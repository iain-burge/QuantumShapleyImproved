
import time
from typing import Union

from qiskit import QuantumCircuit as qc, QuantumRegister as qr
from qiskit.circuit.quantumregister import Qubit as qb
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import Statevector

import numpy as np

from matplotlib import pyplot as plt

from QuantumCORDIC import bitsToIntList, intListToBits, intToBits,\
    dToTheta, qCleanCORDIC, twosCompToInt
from QuantumVotingGame import classicalVoteShapValues, generateVotingGame,\
    voteGate


def sqrShap1b(
        partition: Union[qr, list[qb]], players: Union[qr, list[qb]]
    ) -> qc:
    circuit = qc(partition, players)
    for player in players:
        for i, point in enumerate(partition[::-1]):
            theta = 2*np.arctan(2**(-i-1))
            circuit.cry(-4*theta, point, player) 
            circuit.ry(  2*theta, player) 
    # circuit.draw(output='mpl', style='bw')
    # plt.show()
    return circuit

def shap1b(partition: Union[qr, list[qb]], players: Union[qr, list[qb]]) -> qc:
    circuit = qc(partition, players)
    for player in players:
        for i, point in enumerate(partition[::-1]):
            theta = np.arctan(2**(-i-1))
            circuit.cry(-4*theta, point, player) 
            circuit.ry(  2*theta, player) 
        circuit.ry(np.pi/2, player)
    # circuit.draw(output='mpl', style='bw')
    # plt.show()
    return circuit

def testDistribution():
    n = 1
    l = 4
    
    tReg    = qr(l,   name="t")
    xReg    = qr(l,   name="x")
    yReg    = qr(l,   name="y")
    multReg = qr(l,   name="mult")
    dReg    = qr(l-1, name="d")
    players = qr(n,   name="players")

    cordic = qc(tReg, xReg, yReg, multReg, dReg, players)
    shap   = qc(tReg, xReg, yReg, multReg, dReg, players)
    cordic.append(
        qCleanCORDIC(tReg, xReg, yReg, multReg, dReg, False),
        tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:]
    )
    shap.append(shap1b(dReg, players), dReg[:] + players[:])

    input = []
    expc  = []
    pred  = []

    state = Statevector.from_label(intListToBits(
        [0, 0, 0, 0, 0, 0], [l, l, l, l, l-1, n]
    ))
    state  = state.evolve(cordic)
    state  = state.evolve(shap)
 
    threshold = np.average(list(state.probabilities_dict().values()))
    for key, val in state.probabilities_dict().items():
        # if val < threshold: continue
        print(key)
        if bitsToIntList(key, 4*[l]+[l-1]+[n])[-1] == 0: continue

        input.append(
            twosCompToInt(bitsToIntList(key, 4*[l]+[l-1]+[n])[0], l)
        )
        d = intToBits(
            bitsToIntList(key, 4*[l]+[l-1]+[n])[-2], l-1
        )
        # pred.append(val)
        # pred.append(np.sign(float(state[key]))*val)
        pred.append(float(state[key]))
        expc.append(dToTheta(d))
        print('\t',float(state[key]))
        print(f'\t{dToTheta(d)}')

    # for i in range(1<<l):
    #     print(f"i={i:0{l}b}"
    #           +f'|\t|{np.abs(state.inner(truthy)):.3f}'
    #           +f'|\t|{stateToStr(state)}'
    #     )
    #     input.append(f"{i:0{l}b}")
    #     # expc.append(dToTheta(f"{i:0{l}b}"[::-1]))
    #     pred.append(float(f'{np.abs(state.inner(truthy)):.3f}'))

    expc = np.array(expc)/2 + np.pi/4
    expc = np.sin(expc)/np.linalg.norm(np.sin(expc))
    pred = np.array(pred)/np.linalg.norm(pred)

    plt.scatter(input, expc, label="Expected") 
    plt.scatter(input, pred, label="Predicted") 
    plt.xticks(rotation=60)
    plt.legend()
    plt.show()



def main():
    l          = 5
    
    #Init Registers
    tReg    = qr(l,   name="t")
    xReg    = qr(l,   name="x")
    yReg    = qr(l,   name="y")
    multReg = qr(l,   name="mult")
    dReg    = qr(l-1, name="d")
    allReg  = tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:]

    players = xReg
    votes   = yReg
    aux     = multReg

    utility = votes[-1]

    qubitIndDict = {qubit: i for i, qubit in enumerate(allReg)}

    #Construct Game
    voteWeights, threshold = generateVotingGame(n_players=l, n_votebits=l)
    voteWeights = [5,4,3,2,1] #temp
    threshold   = 8 #temp
    print(time.ctime())
    print(f'{voteWeights, threshold = }')

    #init quantum state
    init_state = Statevector.from_label(intListToBits(
        [0, 0, 0, 0, 0], [l, l, l, l, l-1]
    ))

    #Step 1a
    shap_1a = qc(allReg)
    shap_1a.append(
        qCleanCORDIC(tReg, xReg, yReg, multReg, dReg, False),
        tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:]
    )
    state_1a = init_state.evolve(shap_1a)


    for player_ind in range(l):
        print('\n'+50*'='+'\n'+time.ctime())
        #Defining Player specific registers
        player  = players[player_ind]
        others  = players[:player_ind] + players[player_ind+1:]

        #Step 1b
        shap_1b = qc(allReg)
        shap_1b.append(shap1b(dReg, others), dReg[:] + others[:])
        state_1b = state_1a.evolve(shap_1b)


        #Step 2
        #Init circuits
        shapP  = qc(tReg, xReg, yReg, multReg, dReg)
        shapM  = qc(tReg, xReg, yReg, multReg, dReg)
        shapM.append(
            voteGate(voteWeights, threshold, n_players=l, n_votebits=l),
            players[:] + votes[:] + aux[:]
        )
        shapP.x(player)
        shapP.append(
            voteGate(voteWeights, threshold, n_players=l, n_votebits=l),
            players[:] + votes[:] + aux[:]
        )

        # #Draw circuit
        # shapP.draw('mpl', 0.8, style='bw')
        # shapM.draw('mpl', 0.8, style='bw')
        # plt.show()

        #Bad Step 3
        #  run the circuit
        
        stateP = state_1b.evolve(shapP)
        stateM = state_1b.evolve(shapM)

        pred = stateP.probabilities([qubitIndDict[utility]])[1]\
                -stateM.probabilities([qubitIndDict[utility]])[1]
        print(f'{player_ind=}:',
            f'\n\texpc:\t{classicalVoteShapValues(
                threshold, player_ind, n_players=l, weights=voteWeights
                ):.4f}',
            f'\n\tpred:\t{pred:.4f}'
        )


if __name__=='__main__':
    main()

