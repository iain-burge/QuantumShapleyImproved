
from tqdm import tqdm, trange
import time
from typing import Union

from qiskit import QuantumCircuit as qc, QuantumRegister as qr
from qiskit.circuit.quantumregister import Qubit as qb
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import Statevector

import numpy as np
from functools import cache

from matplotlib import pyplot as plt



#=============================================================================#
#=============================================================================#
circuitDarkMode = {
    "backgroundcolor": "#222222",
    "linecolor": "#FFFFFF",
    "textcolor": "#FFFFFF",
    "gatetextcolor": "#FFFFFF",
}

style = {
    "name": "bw",
    "margins": [0,0,0,0]
}


#=============================================================================#
#=============================================================================#
# Addition Circuit
def additionGate(
        xReg: Union[qr, list[qb]], yReg: Union[qr, list[qb]]
    ) -> Gate:
    """Adds right shifted y to x register. Respects two's complement.
        Based on the simplest version of an algorithm from:
            Quantum Addition Circuits and Unbounded Fan-Out
            Authors: Yasuhiro Takahashi, Seiichiro Tani, Noboru Kunihiro

    Args:
        xReg   (qr): Register to be added to
        yReg   (qr): Register which has input
        rshift (int, optional): The amount right shifted, non-negative only. 
            Defaults to 0.

    Returns:
        Gate: Gate for the addition transformation
    """
    n       = min(len(xReg), len(yReg))
    circuit = qc(xReg, yReg, 
        name=f"({rName(xReg)} + {rName(yReg)}, {rName(yReg)})")

    #Step 1
    for i in range(1,n):
        circuit.cx(yReg[i], xReg[i])
    
    #Step 2
    for i in range(n-2, 0, -1):
        circuit.cx(yReg[i], yReg[i+1])

    #Step 3
    for i in range(n-1):
        circuit.ccx(yReg[i], xReg[i], yReg[i+1])

    #Step 4
    for i in range(n-1, 0, -1):
        circuit.cx(yReg[i], xReg[i])
        circuit.ccx(yReg[i-1], xReg[i-1], yReg[i])

    #Step 5
    for i in range(1, n-1):
        circuit.cx(yReg[i], yReg[i+1])
    
    #Step 6
    for i in range(n):
        circuit.cx(yReg[i], xReg[i])

    return circuit.to_gate()


def shiftAdditionGate(
        xReg: qr, yReg: qr, rshift: int = 1
    ) -> Gate:
    """Adds right shifted y to x register. Respects two's complement.
        Based on the simplest version of an algorithm from:
            Quantum Addition Circuits and Unbounded Fan-Out
            Authors: Yasuhiro Takahashi, Seiichiro Tani, Noboru Kunihiro
        Modified to work with right bitshift

    Args:
        xReg   (qr): Register to be added to
        yReg   (qr): Register which has input
        rshift (int, optional): The amount right shifted, positive values only. 
            Defaults to 1.

    Returns:
        Gate: Gate for the desired transformation
    """
    circuit = qc(xReg, yReg,
        name=f"({rName(xReg)} + {rName(yReg)} / {1<<rshift}, {rName(yReg)})")


    circuit.append(
        additionGate(xReg, yReg),
        xReg[:] + yReg[rshift:]+yReg[:rshift]
    )
    circuit.append(
        additionGate(xReg[-rshift:], yReg[-1:]+yReg[:rshift-1]).inverse(),
        xReg[-rshift:] + yReg[-1:] + yReg[:rshift-1]
    )
    circuit.append(
        additionGate(xReg[-rshift:], yReg[:rshift]),
        xReg[-rshift:] + yReg[:rshift]
    )
    
    # print(circuit.draw()) #temp

    return circuit.to_gate()


#=============================================================================#
#=============================================================================#
#Helper functions
def intToBits(value: int, n: int=0) -> str:
    return f"{value:0>{n}b}"[::-1] 

def bitsToInt(value: str) -> int:
    return int(value[::-1], 2)

def intListToBits(values: list[int], n: Union[int, list[int]]) -> str:
    l = []
    if type(n)==int:
        l = [n] * len(values)
    elif type(n)==list:
        l = n

    string = []
    for i, v in enumerate(values):
        string.append(f"{v:0>{l[i]}b}"[::-1])

    return ''.join(string)[::-1]

def bitsToIntList(string: str, n: Union[int, list[int]]) -> list[int]:
    string = string[::-1]
    values = []

    l = []
    if type(n)==int:
        l = [n] * (len(string)//n)
    elif type(n)==list:
        l = n

    index, next = 0, 0
    for val in l:
        next += val 
        values.append(int(string[index:next][::-1], 2))
        index = next
    
    return values

def stateToStr(value: Statevector) -> str:
    return value.draw('latex_source').replace(r'\rangle', '>')

def stateToIntList(
        value: Statevector, n: Union[int, list[int]], signed=False
    ) -> list[int]:
    string = "".join(filter(str.isdigit, stateToStr(value)))

    #TODO take the sum of list n. Then break string into a list of str
    #with length n, that will be each of our strings
    l = []
    if type(n)==int:
        l = [n] * (len(string)//n)
    elif type(n)==list:
        l = n
    ints = bitsToIntList(string, l)
    if signed:
        ints = [twosCompToInt(val, l[i]) for i, val in enumerate(ints)]
    return ints

def twosCompToInt(value: int, n: int) -> int:
    return value if value<(1<<(n-1)) else value-(1<<n)

def intToTwosComp(value: int, n: int) -> int:
    return value+(1<<n) if value<0 else value

def rName(register) -> str:
    """ Get the register's name. Here because the type hints are messed up
    with qiskit.
        Register (Union[qr, list[qb]]): quantum register

    Returns (str): register's name
    """
    return register[0]._register.name

#=============================================================================#
#=============================================================================#
@cache
def fib(i:int) -> int: 
    '''
    Fibonacci numbers { 1,1,3,5,8,13, ... }

    i (int): Index of Fibonacci number

    return (int): i^th Fibonacci number
    '''
    return int(np.round(
        1/(5**.5) * (((1+5**.5)/2)**(i+1)-((1-5**.5)/2)**(i+1))
    ))


def multGate(
        xReg: qr, aux: qr, m: int, gate: bool = True
    ) -> Union[Gate, qc]:
    '''
    In place multiplication integer by 1+2^(-m) with some error depending on 
    how clean the auxiliary register is.
    Note: the aux register can gain a little error with each use

    xReg     (qr): Register being multiplied
    aux      (qr): Auxiliary register, used to bounce x around
    m       (int): xReg is multiplied by 1+2^(-m)

    Return:
        Gate: Circuit to perform the operation
    '''
    n = len(xReg)

    circuit = qc(xReg, aux, name=f"{rName(xReg)}(1+1/{1<<m})")

    numIter = 2*int(2*np.ceil(np.log(np.sqrt(5)*n/m)/np.log((1+5**.5))/2))

    circuit.append(
        additionGate(aux, xReg),
        aux[:] + xReg[:]
    )
    if not gate: circuit.barrier(label='init_aux')
       
    for i in range(numIter, -1, -1):
        if m*fib(i) >= n: continue

        negative = (fib(i)%2==1)
        if i%2 == 0:
            circuit.append(
                shiftAdditionGate(xReg, aux, m*fib(i)) if negative
                else shiftAdditionGate(xReg, aux, m*fib(i)).inverse(),
                xReg[:] + aux[:]
            )
            if not gate: circuit.barrier(label=f'bounce(aux->x){i}')
        else:
            circuit.append(
                shiftAdditionGate(aux, xReg, m*fib(i)) if negative
                else shiftAdditionGate(aux, xReg, m*fib(i)).inverse(),
                aux[:] + xReg[:]
            )
            if not gate: circuit.barrier(label=f'bounce(x->aux){i}')

    circuit.append(
        additionGate(aux, xReg).inverse(),
        aux[:] + xReg[:]
    )
    if not gate: circuit.barrier(label='clear_aux')

    if gate: return circuit.to_gate()
    else: return circuit

#=============================================================================#
#=============================================================================#
#Little experiment to see what happens to garbage in mult aux register
def garbagePropSim(n: int = 8):
    arrL = []
    arrR = [0]

    for i in range(n):
        if i%2 == 0:
            arrR += [el + fib(i) for el in arrL]
        else:
            arrL += [el + fib(i) for el in arrR]

    print(arrL)
    print(arrR)

    # y=[arrR[i+1] - arrR[i] for i in range(len(arrR)-1)]
    # plt.scatter([i for i in range(len(arrR)-1)], y)
    # y=[arrL[i+1] - arrL[i] for i in range(len(arrL)-1)]
    # plt.scatter([i for i in range(len(arrL)-1)], y)
    plt.scatter([i for i in range(len(arrL))], arrL)
    plt.scatter([i for i in range(len(arrR))], arrR)
    plt.show()



#=============================================================================#
#=============================================================================#
def quantumCORDIC(
        tReg: qr, xReg: qr, yReg: qr, multReg: qr, dReg: qr, gate: bool = True
    ) -> Union[Gate, qc]:
    """ Takes 5 registers of size n, where x,y,mult are zeroed auxiliary,
    t is the input, and d implicitly encodes the output arcsin(t/2^n)
        tReg    (qr):
        xReg    (qr):
        yReg    (qr):
        multReg (qr):
        dReg    (qr):

    Returns:
        Gate:
    """
    n = len(tReg)
    circuit = qc(tReg, xReg, yReg, multReg, dReg, name="asin(t)")

    #Set x = 1 in two's compliment fixed point for range=[-2,2)
    circuit.x(xReg[:-1])


    if not gate: circuit.barrier(label="init_x") #temp

    #CORDIC Iterations
    for i in range(1, n):

        #Infer next rotation direction
        circuit.append(
            # additionGate(tReg, yReg).control(ctrl_state=0).inverse(),
            # xReg[-1:] + tReg[:] + yReg[:]
            additionGate(tReg, yReg).inverse(),
            tReg[:] + yReg[:]
        )
        # circuit.barrier(label="t-y") #temp
        circuit.ccx(xReg[-1], yReg[-1], dReg[-i])
        circuit.ccx(xReg[-1], tReg[-1], dReg[-i])
        circuit.cx(xReg[-1], dReg[-i])
        circuit.cx(tReg[-1], dReg[-i])
        circuit.append(
            # additionGate(tReg, yReg).control(ctrl_state=0),
            # xReg[-1:] + tReg[:] + yReg[:]
            additionGate(tReg, yReg),
            tReg[:] + yReg[:]
        )
        if not gate: circuit.barrier(label=f"calc_d{i}") #temp


        #Perform rotation
        # reflect depending on rotation direction
        for j in range(n):
            circuit.cx(dReg[-i], yReg[j])
            # circuit.cswap(dReg[-i], xReg[j], yReg[j])

        if not gate:
            circuit.barrier(label=f"ref_y{i} ") #temp
        # do the actual rotation
        for j in range(2):
            circuit.append(
                shiftAdditionGate(xReg, yReg, i).inverse(),
                xReg[:] + yReg[:]
            )
            if not gate: circuit.barrier(label=f"rot_A{i}") #temp
            if j == 0:
                circuit.append(
                    multGate(yReg, multReg, 2*i).inverse(),
                    yReg[:] + multReg[:]
                )
            else:
                circuit.append(
                    multGate(yReg, multReg, 2*i),
                    yReg[:] + multReg[:]
                )
            if not gate: circuit.barrier(label=f"rot_B{i}") #temp
            circuit.append(
                shiftAdditionGate(yReg, xReg, i),
                yReg[:] + xReg[:] 
            )
            if not gate: circuit.barrier(label=f"rot_C{i}") #temp


        # undo reflection depending on rotation direction
        for j in range(n):
            circuit.cx(dReg[-i], yReg[j])
            # circuit.cswap(dReg[-i], xReg[j], yReg[j])

        if not gate: circuit.barrier(label=f"unref_y{i}") #temp

        # #Compensate for imperfect rotation
        # circuit.append(
        #     multGate(tReg, multReg, 2*i),
        #     tReg[:] + multReg[:]
        # )

        # if not gate: circuit.barrier(label=f"updt_t{i}") #temp

    if gate: return circuit.to_gate()
    else:    return circuit

def qCleanCORDIC(
        tReg: qr, xReg: qr, yReg: qr, multReg: qr, dReg: qr, gate: bool = True
    ) -> Union[Gate, qc]:
    """ Takes 5 registers of size n, where x,y,mult are zeroed auxiliary,
    t is the input, and d implicitly encodes the output arcsin(t/2^n)
        tReg    (qr):
        xReg    (qr):
        yReg    (qr):
        multReg (qr):
        dReg    (qr):

    Returns:
        Gate:
    """
    circuit = qc(tReg, xReg, yReg, multReg, dReg) 
    circuit.h(tReg[:])
    circuit.x(xReg[:-1])
    circuit.append(
        additionGate(tReg, xReg).inverse(),
        tReg[:] + xReg[:]
    )
    circuit.x(xReg[-2])
    circuit.append(
        quantumCORDIC(tReg, xReg, yReg, multReg, dReg, False),
        tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:]
    )
    circuit.append(
        repairCORDIC(tReg, xReg, yReg, multReg, dReg, False),
        tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:]
    )

    if gate:
        return circuit.to_gate()
    else:
        return circuit



def repairCORDIC(
        tReg: qr, xReg: qr, yReg: qr, multReg: qr, dReg: qr, gate: bool = True
    ) -> Union[Gate, qc]:
    circuit = qc(tReg, xReg, yReg, multReg, dReg)
    circuit.append(
        invRepairCORDIC(tReg, xReg, yReg, multReg, dReg, gate).inverse(),
        tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:]
    )
    if gate:
        return circuit.to_gate()
    else:
        return circuit
    
 
def invRepairCORDIC(
        tReg: qr, xReg: qr, yReg: qr, multReg: qr, dReg: qr, gate: bool = True
    ) -> Union[Gate, qc]:
    """ Takes 5 registers of size n, where x,y,mult are zeroed auxiliary,
    t is the input, and d implicitly encodes the output arcsin(t/2^n)
        tReg    (qr):
        xReg    (qr):
        yReg    (qr):
        multReg (qr):
        dReg    (qr):

    Returns:
        Gate:
    """
    n = len(tReg)
    circuit = qc(tReg, xReg, yReg, multReg, dReg, name="asin(t)")

    #Set x = 1 in two's compliment fixed point for range=[-2,2)
    circuit.x(xReg[-2])


    if not gate: circuit.barrier(label="init_x") #temp

    #CORDIC Iterations
    for i in range(1, n):
        # #Infer next rotation direction
        # circuit.append(
        #     # additionGate(tReg, yReg).control(ctrl_state=0).inverse(),
        #     # xReg[-1:] + tReg[:] + yReg[:]
        #     additionGate(tReg, yReg).inverse(),
        #     tReg[:] + yReg[:]
        # )
        # # circuit.barrier(label="t-y") #temp
        # circuit.ccx(xReg[-1], yReg[-1], dReg[-i])
        # circuit.ccx(xReg[-1], tReg[-1], dReg[-i])
        # circuit.cx(xReg[-1], dReg[-i])
        # circuit.cx(tReg[-1], dReg[-i])
        # circuit.append(
        #     # additionGate(tReg, yReg).control(ctrl_state=0),
        #     # xReg[-1:] + tReg[:] + yReg[:]
        #     additionGate(tReg, yReg),
        #     tReg[:] + yReg[:]
        # )
        # if not gate: circuit.barrier(label=f"calc_d{i}") #temp


        #Perform rotation
        # reflect depending on rotation direction
        for j in range(n):
            circuit.cx(dReg[-i], yReg[j])
            # circuit.cswap(dReg[-i], xReg[j], yReg[j])

        if not gate:
            circuit.barrier(label=f"ref_y{i} ") #temp
        # do the actual rotation
        for j in range(2):
            circuit.append(
                shiftAdditionGate(xReg, yReg, i).inverse(),
                xReg[:] + yReg[:]
            )
            if not gate: circuit.barrier(label=f"rot_A{i}") #temp
            if j == 0:
                circuit.append(
                    multGate(xReg, multReg, 2*i).inverse(),
                    xReg[:] + multReg[:]
                )
            else:
                circuit.append(
                    multGate(yReg, multReg, 2*i),
                    yReg[:] + multReg[:]
                )
            if not gate: circuit.barrier(label=f"rot_B{i}") #temp
            circuit.append(
                shiftAdditionGate(yReg, xReg, i),
                yReg[:] + xReg[:] 
            )
            if not gate: circuit.barrier(label=f"rot_C{i}") #temp


        # undo reflection depending on rotation direction
        for j in range(n):
            circuit.cx(dReg[-i], yReg[j])
            # circuit.cswap(dReg[-i], xReg[j], yReg[j])

        if not gate: circuit.barrier(label=f"unref_y{i}") #temp

        # #Compensate for imperfect rotation
        # circuit.append(
        #     multGate(tReg, multReg, 2*i),
        #     tReg[:] + multReg[:]
        # )

        if not gate: circuit.barrier(label=f"updt_t{i}") #temp

    if gate: return circuit.to_gate()
    else:    return circuit


def dToTheta(d: str) -> float:
    theta = 0
    
    # print(d)
    for i, dir in enumerate(d[::-1]):
        # print(f"\t{(i,dir)=}")
        theta += (-1 if dir=='1' else 1) * 2 * np.arctan(2**(-i-1))

    return theta

#=============================================================================#
#=============================================================================#
#Debug
def debugCORDIC(t: int, n_bits: int):
    tReg    = qr(n_bits, name="t")
    xReg    = qr(n_bits, name="x")
    yReg    = qr(n_bits, name="y")
    multReg = qr(n_bits, name="mult")
    dReg    = qr(n_bits, name="d")

    cordic  = qc(tReg, xReg, yReg, multReg, dReg)
    for i, bit in enumerate(intToBits(t, n_bits)):
        if bit == '1': cordic.x(i)
    cordic.append(
        quantumCORDIC(tReg, xReg, yReg, multReg, dReg, False),
        tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:]
    )
    cordic  = cordic.decompose()

    sub     = qc(tReg, xReg, yReg, multReg, dReg)

    # cordic.draw(output="latex", style=style, 
    #         interactive=True, fold=80, justify="center",
    #         vertical_compression="high")
    # plt.show()

    inState = intListToBits([0,0,0,0,0], n_bits)
    state = Statevector.from_label(inState)
    print(f"in:\t{stateToIntList(state, n_bits)}")

    for unitary in cordic:
        if "barrier" in unitary.operation.name:
            state = state.evolve(sub)
            print(f"{unitary.operation.label}:"
                  +f"\t{stateToIntList(state, n_bits)}"
                  +f"\t\tGate Depth={sub.decompose(reps=32).depth()}"
            )
            sub.clear()
        else:
            sub.append(unitary)

    print(f"out:\t{dToTheta(intToBits(
        stateToIntList(state, n_bits)[-1], n_bits))}")

def debugShiftAdd(signedx: int, signedy: int, n_bits: int):
    xReg    = qr(n_bits, name="x")
    yReg    = qr(n_bits, name="y")
 
    x, y = intToTwosComp(signedx, n_bits), intToTwosComp(signedy, n_bits)

    print(f"{signedx=}, {signedy=}, {x=}, {y=}")
    shift   = 3
    circuit = qc(xReg, yReg)
    sub     = qc(xReg, yReg)
    inState = intListToBits([x,y], n_bits)
    circuit.append(
        shiftAdditionGate(xReg, yReg, shift),
        xReg[:] + yReg[:]
    )
    state = Statevector.from_label(inState)
    print(f"in:\t\t{stateToIntList(state, n_bits, True)}")
    for unitary in circuit.decompose():
        sub.append(unitary)    
        state = state.evolve(sub)
        print(f"{unitary.operation.name}:"
              +f"\t{stateToIntList(state, n_bits)}"
              +f"\t\tGate Depth={sub.decompose(reps=32).depth()}"
        )
        sub.clear()
    # state = state.evolve(circuit)
    print(f"out:\t\t {
        twosCompToInt(stateToIntList(state,n_bits,True)[0],n_bits)}")
    print(f"expected:\t{(signedx+(signedy*(2**(-shift))),signedy)}")
    return

 
def debugMULT(x: int, rshift: int, n_bits: int):
    xReg    = qr(n_bits, name="x")
    multReg = qr(n_bits, name="mult")

    mult = qc(xReg, multReg)
    mult.append(
        multGate(xReg, multReg, rshift, False),
        xReg[:] + multReg[:]
    )
    mult  = mult.decompose()

    sub     = qc(xReg, multReg)

    inState = intListToBits([x,0], n_bits)
    state = Statevector.from_label(inState)
    print(f"in:\t{stateToIntList(state, n_bits)}")

    for unitary in mult:
        if "barrier" in unitary.operation.name:
            state = state.evolve(sub)
            print(f"{unitary.operation.label}:"
                  +f"\t{stateToIntList(state, n_bits)}"
                  +f"\t\tGate Depth={sub.decompose(reps=32).depth()}"
            )
            sub.clear()
        else:
            sub.append(unitary)

def cordicComplexity(n_max: int) -> None:
    n_val      = []
    opComplexity = []
    cnotCount  = []
    for i in trange(3, n_max):
        try:
            tReg=qr(i); xReg=qr(i); yReg=qr(i); multReg=qr(i); dReg=qr(i-1)
            allReg  = tReg[:]+xReg[:]+yReg[:]+multReg[:]+dReg[:]
            circuit = qc(allReg)
            circuit.append(
                quantumCORDIC(tReg, xReg, yReg, multReg, dReg, False), allReg
            )
            # cxs = (circuit.decompose(reps=32).count_ops())['cx']
            # cnotCount.append(cxs)
            # opComplexity.append(circuit.decompose(reps=32).depth()-cxs)
            n_val.append(i)
        except KeyboardInterrupt:
            break
    plt.title('CORDIC ASIN Complexity')
    plt.scatter(n_val, opComplexity, label='Single Qubit Gates', marker='s')
    plt.scatter(n_val, cnotCount, label='CNOT Gates', marker='D')
    plt.legend()
    plt.show()



#=============================================================================#
#=============================================================================#
def main():
    # cordicComplexity(32); return #temp
    n_bits  = 4

    tReg    = qr(n_bits, name="t")
    xReg    = qr(n_bits, name="x")
    yReg    = qr(n_bits, name="y")
    multReg = qr(n_bits, name="mult")
    dReg    = qr(n_bits, name="d")

    debugCORDIC(3, n_bits)
    allReg  = tReg[:] + xReg[:] + yReg[:] + multReg[:] + dReg[:]

    cordic = qc(tReg, xReg, yReg, multReg, dReg)
    cordic.append(
        qCleanCORDIC(tReg, xReg, yReg, multReg, dReg, False), allReg
    )

    inputs  = []
    outputs = []

    print(f"start time: {time.ctime()}")
    state   = Statevector.from_label(
            intListToBits([0,0,0,0,0], n_bits))
    state = state.evolve(cordic)
    print(f"end time:  {time.ctime()}")

    # sub    = qc(tReg, xReg, yReg, multReg, dReg)
    # width = 1
    # for i, op in enumerate(tqdm(cordic.decompose(reps=32))):
    #     sub.append(op)
    #     if i%width==0:
    #         state = state.evolve(sub)
    #         sub.clear()
    # if i%width!=0:
    #     state = state.evolve(sub)
    #     sub.clear()

    threshold = np.average(list(state.probabilities_dict().values()))
    print(state.probabilities_dict())
    for key, val in state.probabilities_dict().items():
        if val < threshold: continue
        print(bitsToIntList(key, n_bits))
        inputs.append(
            twosCompToInt(bitsToIntList(key, n_bits)[0],n_bits)
        )
        # print(stateToIntList)
        d = intToBits(
            bitsToIntList(key, n_bits)[-1], n_bits
        )       
        outputs.append(dToTheta(d))
        # print(key)
        # print(f"{bitsToIntList(key, 4*[n_bits]+[n_bits-1])}|"
              # +f"\tsigned t={twosCompToInt(t,n_bits)}|\t {d=}")
    

    # print(f"start time: {time.ctime()}")
    # print(f"for j in range({1<<(n_bits-1)}):")
    # for j in range(1<<(n_bits-1)):
    #     t = intToTwosComp(twosCompToInt(j, n_bits-1), n_bits)
    #     state   = Statevector.from_label(
    #             intListToBits([t,0,0,0,0], 4*[n_bits]+[n_bits-1]))
    #     final   = state.evolve(cordic)
    #     # print(2**n_bits*np.arcsin(t/(2**(n_bits)))/(2**(n_bits)))
    #     # final   = Statevector.from_label(intListToBits([t,0,0,0,t], n_bits)) #temp
    #     inputs.append(
    #         twosCompToInt(t,n_bits), 
    #     )
    #     # print(stateToIntList)
    #     d = intToBits(
    #         stateToIntList(final, 4*[n_bits]+[n_bits-1])[-1], n_bits-1
    #     )
    #     outputs.append(dToTheta(d))
    #     print(f"{(j,t)=}|\tsigned t={twosCompToInt(t,n_bits)}|\t {d=}|\t {time.ctime()}")

        
    # cordic.draw(output="latex", style=style, 
    #         interactive=True, fold=80, justify="center",
    #         vertical_compression="high")
    # plt.show()


    inRange   = np.array(inputs)
    expected  = np.arcsin(inRange/(2**(n_bits-1)))
    predicted = np.array(outputs)
    print([f'{val:.2f}' for val in outputs])
    print([(inputs[i],outputs[i]) for i in range(len(inputs))])
    print(f"{np.max(np.abs(expected-predicted))             = :.6f}")
    print(f"{inRange[np.argmax(np.abs(expected-predicted))] = }")
    print(f"{np.mean(np.abs(expected-predicted))            = :.6f}")
    print(f"{np.median(np.abs(expected-predicted))          = :.6f}")
    plt.scatter(inRange, expected,  label="Expected")
    plt.scatter(inRange, predicted, label="Predicted")
    plt.title("Quantum CORDIC Approx")
    plt.legend()
    # plt.xticks(inRange)
    # plt.yticks(outRange/(2**(n_bits+1)))
    plt.grid()
    plt.show()

    # debugCORDIC(9, n_bits)
    # debugMULT(4, 2, n_bits)
    # findBadMult(n_bits, 1)



if __name__ == "__main__":
    main()
    
#=============================================================================#
#=============================================================================#

