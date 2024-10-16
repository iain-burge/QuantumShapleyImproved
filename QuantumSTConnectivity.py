
from collections.abc import Callable
import time
from tqdm import tqdm, trange
from typing import Union

from qiskit import QuantumCircuit as qc, QuantumRegister as qr
from qiskit.circuit.library import QFT, StatePreparation, ZGate
from qiskit.circuit.quantumregister import Qubit as qb
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import Statevector

import numpy as np

from matplotlib import pyplot as plt

from QuantumCORDIC import bitsToIntList, intListToBits, intToBits,\
    dToTheta, qCleanCORDIC, twosCompToInt
from QuantumVotingGame import classicalVoteShapValues, gamma, generateVotingGame,\
    voteGate

def earlyShap1a(ptReg: qr) -> qc:
    circuit = qc(ptReg, name='shap1a')

    l   = len(ptReg); L = 1<<l
    w_l = np.array([
        np.sqrt(0.5*np.sin(np.pi/L)*np.sin(np.pi*(k+1)/L))
        # np.sqrt(np.sin(np.pi*(k+1)/(2*L))**2 - np.sin(np.pi*(k)/(2*L))**2)
        for k in range(L)
    ], dtype=float)
    w_l /= np.linalg.norm(w_l)

    circuit.append(StatePreparation(list(w_l)), ptReg[:])

    return circuit


def earlyShap1b(ptReg: qr, plReg: Union[qr, list[qb]]) -> qc:
    circuit = qc(ptReg, plReg, name='shap1b')
    
    #TODO implement stuff lmao
    for player in plReg:
        for j, point in enumerate(ptReg[::-1]):
            circuit.cry((np.pi/2)/(2**j), point, player)
        circuit.ry((np.pi/2)/(2**len(ptReg)),player)

    return circuit

def tableCircuit(plReg: qr, utReg: qr, f: Callable[[str], bool]) -> qc:
    circuit = qc(plReg, utReg, name='f(x)')

    n = len(plReg); N = 1<<n
    for i in range(N):
        bits = intToBits(i, n)
        if f(bits):
            circuit.mcx(plReg[:], utReg, ctrl_state=bits)

    return circuit

def monteCarloShap(
        n_players: int, target: int, numSamples: int, f: Callable[[str], bool]
    ) -> float:
    avg = 0
    players = [i for i in range(n_players) if i != target]

    for _ in range(numSamples):
        subsize = np.random.randint(n_players)
        subset  = np.random.choice(players, subsize, replace=False)

        subbits = n_players * ['0']
        for player in subset: subbits[player] = '1'
        subbitsN = subbits.copy()
        subbitsP = subbits.copy(); subbitsP[target] = '1'
        # print(subsize, subset, subbits)

        avg += int(f(''.join(subbitsP))) - int(f(''.join(subbitsN)))

    avg /= numSamples

    return avg


def trueShap(n_players: int, target: int, f: Callable[[str], bool]) -> float:
    shap = 0
    for i in range(1<<n_players):
        bits = intToBits(i, n_players)
        m = (bits[:target] + bits[target+1:]).count('1')

        value = int(f(''.join(bits)))
        shap += (-1 if bits[target]=='0' else 1) * gamma(n_players-1, m) * value

    return shap


def Qcircuit(
        ptReg: qr, plReg: qr, utReg: qr, target: int, f: Callable[[str], bool],
        barriers: bool = False
    ) -> qc:
    targQubit = plReg[target]
    otherReg  = plReg[:target] + plReg[target+1:]

    circuit = qc(ptReg, plReg, utReg, name='Q')

    #Flip Phase
    if barriers: circuit.barrier(label='Flip Phase')
    circuit.p(np.pi, utReg)


    #f(x) reflection
    if barriers: circuit.barrier(label='S_{f(x)} Reflection')

    circuit.x(targQubit)
    circuit.append(tableCircuit(plReg, utReg, f).to_gate(), plReg[:] + utReg[:])
    circuit.x(targQubit)
    circuit.append(tableCircuit(plReg, utReg, f).to_gate(), plReg[:] + utReg[:])

    circuit.z(utReg)

    circuit.append(tableCircuit(plReg, utReg, f).to_gate(), plReg[:] + utReg[:])
    circuit.x(targQubit)
    circuit.append(tableCircuit(plReg, utReg, f).to_gate(), plReg[:] + utReg[:])
    circuit.x(targQubit)

    # #A^(-1)
    if barriers: circuit.barrier(label='A^{-1}')
    circuit.append(earlyShap1b(ptReg, otherReg).to_gate().inverse(), 
                   ptReg[:] + otherReg[:])
    circuit.append(earlyShap1a(ptReg).to_gate().inverse(), ptReg[:])

    #Reflection wrt 0
    if barriers: circuit.barrier(label='S_0 Reflection')
    circuit.x(ptReg[:]+plReg[:]+utReg[:])
    circuit.append(
        ZGate().control(len(ptReg)+len(plReg)),
        ptReg[:]+plReg[:]+utReg[:]
    )
    circuit.x(ptReg[:]+plReg[:]+utReg[:])

    #A
    if barriers: circuit.barrier(label='A')
    circuit.append(earlyShap1a(ptReg).to_gate(), ptReg[:])
    circuit.append(earlyShap1b(ptReg, otherReg).to_gate(), ptReg[:] + otherReg[:])

    return circuit


def testShap(n: int, l: int, aux: int = 0) -> None:
    #Init Registers
    ptReg  = qr(l, name='Partition')
    plReg  = qr(n, name='Player')
    auxReg = qr(aux, name='Aux')
    utReg  = qr(1, name='Utility')

    #Init State
    state = Statevector.from_label((l+n+aux+1)*'0')

    allReg       = ptReg[:] + plReg[:] + auxReg[:] + utReg[:]
    qubitIndDict = {qubit: i for i, qubit in enumerate(allReg)}
    def f(bits: str) -> bool:
            b = [bit=='1' for bit in bits[::-1]]
            return (b[0] and b[1]) or (b[2] and b[4])\
                or (b[2] and b[5] and b[6])


    for target in range(n):
        #Init circuit
        circuit = qc(ptReg, plReg, auxReg, utReg)

        #Denote target
        targetReg = plReg[target]
        otherReg  = plReg[:target] + plReg[target+1:]

        #Step 1 a and b
        circuit.append(earlyShap1a(ptReg), ptReg[:])
        circuit.append(earlyShap1b(ptReg, otherReg), ptReg[:]+otherReg[:])

        #Step 2+
                # f = lambda bits: bits[::-1] in ['110', '101', '111']
        circuitp = circuit.copy()
        circuitp.x(plReg[target])
        circuitp.append(
            tableCircuit(plReg, utReg, f),
            plReg[:] + utReg[:]
        )

        circuitn = circuit.copy()
        circuitn.append(
            tableCircuit(plReg, utReg, f),
            plReg[:] + utReg[:]
        )

        # circuitp.draw('mpl', style='bw')
        # # tableCircuit(plReg, utReg, f).draw('mpl', style='bw')
        # plt.show()

        statep = state.evolve(circuitp)
        staten = state.evolve(circuitn)

        pos = statep.probabilities([qubitIndDict[utReg[0]]])[1]
        neg = staten.probabilities([qubitIndDict[utReg[0]]])[1]

        expected = trueShap(n, n-target-1, f)
        quantum  = pos - neg
        mcMethod = monteCarloShap(n,n-target-1,1<<(2*l),f)
        print(f'{target=}')
        print(f'\tshap       = {expected:.5f}')
        print(f'\tqmc method = {quantum:.5f}\t(err={abs(quantum-expected):.5f})')
        print(f'\tmc method  = {mcMethod:.5f}\t(err={abs(mcMethod-expected):.5f})')

    # print(state)
    # plt.plot(state)
    # plt.show()




def main():
    est = 5
    l   = 4
    n   = 7
    # n   = 3
    aux = 0


    #Init Registers
    estReg = qr(est, name='Amp Est')
    ptReg  = qr(l, name='Partition')
    plReg  = qr(n, name='Player')
    auxReg = qr(aux, name='Aux')
    utReg  = qr(1, name='Utility')


    allReg       = estReg[:] + ptReg[:] + plReg[:] + auxReg[:] + utReg[:]
    qubitIndDict = {qubit: i for i, qubit in enumerate(allReg)}

    def f(bits: str) -> bool:
            b = [bit=='1' for bit in bits[::-1]]
            return (b[0] and b[1]) or (b[2] and b[4])\
                or (b[2] and b[5] and b[6])

    # f = lambda bits: bits[::-1] in ['110', '101', '111']

    # testShap(n, l)
    # Qcircuit(ptReg, plReg, utReg, 0, f).draw('mpl', filename='/home/iain/Downloads/test', style='bw', fold=120)
    # plt.show()

    for target in range(n): #temp
        print(156*'=')
        print(f'{target = }')
        #Init State
        state = Statevector.from_label((est+l+n+aux+1)*'0')

        #Init circuit
        circuit = qc(estReg, ptReg, plReg, auxReg, utReg)
        
        #Init Q-Gate
        QGate   = Qcircuit(ptReg, plReg, utReg, target, f)#.to_gate()
        # QGate.draw('mpl', fold=80, style='bw')
        # plt.show()

        targetReg = plReg[target]
        otherReg  = plReg[:target] + plReg[target+1:]

        print(f'Construct Circuit - {time.ctime()}')
        circuit.barrier(label='Prep')
        circuit.append(QFT(len(estReg)), estReg[:])
        circuit.append(earlyShap1a(ptReg), ptReg[:])
        circuit.append(earlyShap1b(ptReg, otherReg), ptReg[:]+otherReg[:])

        circuit.barrier(label='Amps')
        for i, bit in enumerate(tqdm(estReg)):
            # print(i+1,'/',len(estReg),'\t',time.ctime())
            circuit.append(
                QGate.power(1<<i).control(1),
                [bit] + ptReg[:] + plReg[:] + utReg[:]
            )

        circuit.barrier(label='Processing')
        circuit.append(QFT(len(estReg), inverse=True), estReg[:])
        circuit.x(estReg[-1]) #weird hack fix

        # circuit.decompose().draw(
        #     'mpl', fold=-1, style='bw', 
        #     filename=f'./images/stCon_{time.ctime()}'
        # )
        # plt.show()

        print(f'Evolve State      - {time.ctime()}')
        # circuit.draw('mpl', fold=80, style='bw'); plt.show() #temp
        state = state.evolve(circuit)
        results = np.array(
            state.probabilities([qubitIndDict[bit] for bit in estReg])
        )

        print(f'Display Results   - {time.ctime()}')
        outputs = np.sin56((np.pi/2**est) * np.arange(2**est))**2
        plt.scatter(outputs, results)
        # plt.bar(np.arange(2**est), results, align="edge", width=.9)
        # plt.show()

        # print('fake', np.dot(outputs, results))
        quantumOut  = outputs[np.argmax(results)]
        expectedOut = trueShap(n, n-target-1, f)
        monteCarOut = monteCarloShap(n, n-target-1, 1<<est, f)
        print()
        print(f'\tTrue Value:  {expectedOut:.5f}')
        print(f'\tQuantum:     {quantumOut:.5f}'+
              f'\t(Error={abs(expectedOut-quantumOut):.5f})')
        print(f'\tMonte Carlo: {monteCarOut}'+
              f'\t(Error={abs(expectedOut-monteCarOut):.5f})')
        print()


if __name__ == "__main__":
    main()

