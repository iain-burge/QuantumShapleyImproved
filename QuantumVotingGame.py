
import time
from tqdm import tqdm, trange
from typing import Union

from qiskit import QuantumCircuit as qc, QuantumRegister as qr
from qiskit.circuit.quantumregister import Qubit as qb
from qiskit.circuit.gate import Gate
from qiskit.quantum_info import Statevector

from QuantumCORDIC import additionGate, intListToBits, intToBits, stateToIntList

import numpy as np
from math import comb 
from functools import cache

from matplotlib import pyplot as plt

def gamma(n:int, m:int):
    return 1/((n+1)*comb(n,m))

def classicalVoteShapValues(
        threshold: int, player: int, n_players: int, weights: list[int]
    ) -> float:
    shap = 0

    for i in range(1<<n_players):
        bits = intToBits(i, n_players)
        m = (bits[:player] + bits[player+1:]).count('1')
        tally = 0
        for j in range(n_players):
            if bits[j] == '1': tally+=weights[j]
        value = 1 if tally>=threshold else 0
        shap += (-1 if bits[player]=='0' else 1) * gamma(n_players-1, m) * value
        # print('\texpc:\t',tally,
              # f'\t{"success" if tally>=threshold else "fail"}')

    return shap
   

def generateVotingGame(
        n_players: int, n_votebits: int
    ) -> tuple[list[int],int]:
    voteWeights  = np.random.rand(n_players)
    voteWeights /= sum(voteWeights)

    threshold     = np.random.randint(1<<(n_votebits//2-1), (1<<(n_votebits-1)))
    grandCoaltion = np.random.randint(threshold, (1<<(n_votebits-1)))

    voteWeights *= grandCoaltion
    voteWeights  = np.round(voteWeights)
    voteWeights  = [int(weight) for weight in voteWeights] 

    return voteWeights, threshold


def fixedAdditionGate(
        votes: Union[qr,list[qb]], aux: Union[qr,list[qb]], x: int
    ) -> Gate:
    n_votebits = len(votes)
    circuit = qc(votes, aux, name=f'+{x}')

    for i, bit in enumerate(intToBits(x, n_votebits)):
        if bit == '1': circuit.x(aux[i])
    circuit.append(additionGate(votes, aux), votes[:]+aux[:])
    for i, bit in enumerate(intToBits(x, n_votebits)):
        if bit == '1': circuit.x(aux[i])

    return circuit.to_gate()


def voteGate(
        voteWeights: list[int], threshold: int, n_players: int, n_votebits: int
    ) -> Gate:
    players = qr(n_players,  name='player')
    votes   = qr(n_votebits, name='vote')
    aux     = qr(n_votebits, name='aux')
    circuit = qc(players, votes, aux, name=f'Vote')

    for i, player in enumerate(players):
        circuit.append(
            fixedAdditionGate(votes, aux, voteWeights[i]).control(),
            [player] + votes[:] + aux[:]
        )

    circuit.append(
        fixedAdditionGate(votes, aux, threshold).inverse(),
        votes[:] + aux[:]
    )
    circuit.append(
            fixedAdditionGate(votes[:-1], aux[:-1], threshold),
        votes[:-1] + aux[:-1]
    )
    circuit.x(votes[-1])

    # circuit.draw('mpl', scale=0.8, style='bw')
    # plt.show()

    return circuit.to_gate()

def main():
    n_players  = 3
    n_votebits = 3

    votes   = qr(n_votebits)
    aux     = qr(n_votebits)
    players = qr(n_players)

    voteWeights, threshold = generateVotingGame(n_players, n_votebits)
    voteWeights = [2,1,0]
    threshold = 2
    print(f'{voteWeights, threshold = }')

    # print(classicalVoteShapValues(4, 2, 3, [3,2,1]))

    for i in range(1<<n_players):
        print(intToBits(i, n_players))
        state   = Statevector.from_label(
            intListToBits([i,0,0], [n_players]+2*[n_votebits]))
        circuit = qc(players, votes, aux)
        circuit.append(
            voteGate(voteWeights, threshold, n_players, n_votebits), 
            players[:] + votes[:] + aux[:]
        )
        print('\tinit:\t', stateToIntList(
            state, [n_players]+[n_votebits-1]+[1]+[n_votebits]
        ))
        state = state.evolve(circuit)
        print('\tpred:\t', stateToIntList(
            state, [n_players]+[n_votebits-1]+[1]+[n_votebits]
        ))
        tally = 0
        for j in range(n_players):
            if intToBits(i, n_players)[j] == '1': tally+=voteWeights[j]
        print('\texpc:\t',tally,
              f'\t{"success" if tally>=threshold else "fail"}')

    
if __name__=="__main__":
    main()

