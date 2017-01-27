"""
Created on Sat Nov 19 21:47:01 2016
@author: gopal
"""
import numpy as np
#P(S'={1..81} |S={1..81},a={1,2,3,4})
states = 81; actions = 4; gamma = 0.9925
#P[action][PreviousState][NextState]
def LoadInput():
    P = np.zeros(shape=(actions, states, states))
    for i in range(actions):
        Pa_raw = np.loadtxt('prob_a' + str(i+1) + '.txt')
        for item in Pa_raw:
            P[i][item[0]-1][item[1]-1] = item[2]
    R = np.loadtxt('rewards.txt') 
    return P, R    

P, R = LoadInput()
#StateValue(S) = R(S) + gamma* SumOverS_dash(P[policy(S)][S][S_dash] * StateValue(S_dash))
def EvaluatePolicy(Policy):
    I = np.identity(states, dtype=int)
    P_policy = np.zeros(shape = (states, states));
    for S in range(states):
        for S_dash in range(states):
            P_policy[S][S_dash] = P[Policy[S]][S][S_dash]
    return np.linalg.solve(I - gamma*P_policy, R)
        
def GetBetterPolicy(Policy):
    V = EvaluatePolicy(Policy);
    Policy_better = np.argmax([np.inner(P[i][:][:], V) for i in range(actions)], axis=0)
    return Policy_better

def GetOptimumPolicy(T, startPolicy = 0, stopAtConvergence = False, epsilon=0):
    Policy = np.zeros(states) + startPolicy
    for t in range(T):
        betterPolicy = GetBetterPolicy(Policy)
        if stopAtConvergence and np.count_nonzero(betterPolicy - Policy) == epsilon:
            Policy = betterPolicy
            break;            
        Policy = betterPolicy
    return Policy, EvaluatePolicy(Policy), t
    
def GetBetterStateValues(V):
    V_better = R + gamma*np.max([np.inner(P[i][:][:], V) for i in range(actions)], axis=0)
    return V_better

def GetOptimumStateValues(T, stopAtConvergence = False, epsilon=0):
    Value = np.zeros(states)
    for t in range(T):
        betterValue = GetBetterStateValues(Value)
        if stopAtConvergence and np.max(np.abs(betterValue - Value)) <= epsilon:
            Value = betterValue
            break;
        Value = betterValue
    Policy_star = np.argmax([np.inner(P[i][:][:], Value) for i in range(actions)], axis=0)
    return Policy_star, Value, t 