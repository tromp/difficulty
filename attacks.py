#!/usr/bin/pypy
import random, traceback, platform, sys
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Output, Input, State
from math import *
import json
import time

import mining

sys.setrecursionlimit(100000)

worstcases = {'asert':2.5*log(2),
              'aserti3':2.5*log(2)}

params = {'INITIAL_BCC_BITS':0x18084bb7,
        'INITIAL_TIMESTAMP':1503430225,
        'INITIAL_HEIGHT':481824}

def fast_forward_attack(algoname, forward_blocks=1, total_blocks=500, total_jump=None, FTL=7200, autoend=False):
  algo = mining.Algos[algoname]
  if not total_blocks:
    try:
      algtype = algoname.split('-')[0]
      N = algoname.split('-')[-1]
      if algtype in worstcases:
        total_blocks = int(N * worstcases['algtype'])
      else:
        total_blocks = int(N * 2.5*log(2))
    except:
      pass
  if not total_jump:
    total_jump = total_blocks*mining.IDEAL_BLOCK_TIME + FTL

  states = []
  mining.states = states # ugh
  N = 2020
  for n in range(-N, 0):
    state = mining.State(params['INITIAL_HEIGHT'] + n, params['INITIAL_TIMESTAMP'] + n * mining.IDEAL_BLOCK_TIME,
                         params['INITIAL_TIMESTAMP'] + n * mining.IDEAL_BLOCK_TIME,
                         params['INITIAL_BCC_BITS'], mining.bits_to_work(params['INITIAL_BCC_BITS']) * (n + N + 1),
                         1., 1, 0.0, 0.5, 0.0, False, '')
    states.append(state)

  chainwork = states[-1].chainwork
  expected_work = chainwork + mining.bits_to_work(params['INITIAL_BCC_BITS']) * total_blocks
  n = 0
  profit = bestprofit = 0.
  while chainwork < expected_work or autoend:
    bits = algo.next_bits('', **algo.params)
    target = mining.bits_to_target(bits)
    chainwork = states[-1].chainwork + mining.bits_to_work(bits)
    wall_time = states[-1].wall_time + mining.IDEAL_BLOCK_TIME
    if n < forward_blocks:
      timestamp = states[-1].timestamp + total_jump // forward_blocks
    else:
      timestamp = states[-1].timestamp + 1
    states.append(mining.State(states[-1].height + 1, wall_time, timestamp, bits, chainwork, 1, 0, 1, 0, 0, 0, ''))
    n += 1
    if autoend and n > total_blocks:
      bestprofit = max(profit, bestprofit)
      profit = (1/((chainwork-states[N].chainwork) / (len(states)-N-1) / mining.bits_to_work(params['INITIAL_BCC_BITS'])))
      if chainwork > expected_work and profit < bestprofit:
        del states[-1]
        break
  #print(profit)
  states = states[N-1:]
  profit = (len(states)-1) / total_blocks * (expected_work-states[0].chainwork) / (chainwork-states[0].chainwork)*100 - 100
  diff = mining.bits_to_work(states[-1].bits) / mining.bits_to_work(states[-0].bits)
  return states, len(states)-1, profit, diff

if __name__ == '__main__':
  alg, f, t, j, FTL = 'cw-144', 1, 145, None, 7200
  s, n, p, d = fast_forward_attack(alg, f, t, j, FTL, True)
  print("Alg: %-10s fwd=%-3i tot=%-3i jump=%s, FTL=%i\tBlocks mined: %i\tEnd diff: %3.1f%%\tProfit: %3.1f%%\t" % (alg, f, t, str(j), FTL, n, d*100, p))

  alg, f, t, j, FTL = 'asert-144', 1, 283, None, 7200
  s, n, p, d = fast_forward_attack(alg, f, t, j, FTL)
  print("Alg: %-10s fwd=%-3i tot=%-3i jump=%s, FTL=%i\tBlocks mined: %i\tEnd diff: %3.1f%%\tProfit: %3.1f%%\t" % (alg, f, t, str(j), FTL, n, d*100, p))

  alg, f, t, j, FTL = 'asert-576', 1, 1290, None, 7200
  s, n, p, d = fast_forward_attack(alg, f, t, j, FTL, True)
  print("Alg: %-10s fwd=%-3i tot=%-3i jump=%s, FTL=%i\tBlocks mined: %i\tEnd diff: %3.1f%%\tProfit: %3.1f%%\t" % (alg, f, t, str(j), FTL, n, d*100, p))
  
  alg, f, t, j, FTL = 'wtema-144', 48, 330, None, 7200
  s, n, p, d = fast_forward_attack(alg, f, t, j, FTL, True)
  print("Alg: %-10s fwd=%-3i tot=%-3i jump=%s, FTL=%i\tBlocks mined: %i\tEnd diff: %3.1f%%\tProfit: %3.1f%%\t" % (alg, f, t, str(j), FTL, n, d*100, p))
  alg, f, t, j, FTL = 'wtema-576', 75, 1200, None, 7200
  s, n, p, d = fast_forward_attack(alg, f, t, j, FTL, True)
  print("Alg: %-10s fwd=%-3i tot=%-3i jump=%s, FTL=%i\tBlocks mined: %i\tEnd diff: %3.1f%%\tProfit: %3.1f%%\t" % (alg, f, t, str(j), FTL, n, d*100, p))

  print()
  # optimum attack not yet found, but here's something kinda close
  alg, f, t, j, FTL = 'lwma-144', 1, 100, None, 7200
  for t in range(120, 125, 1):
    s, n, p, d = fast_forward_attack(alg, f, t, j, FTL, True)
    print("Alg: %-10s fwd=%-3i tot=%-3i jump=%s, FTL=%i\tBlocks mined: %i\tEnd diff: %3.1f%%\tProfit: %3.1f%%\t" % (alg, f, t, str(j), FTL, n, d*100, p))
