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

params = {'INITIAL_BCC_BITS':0x18084bb7,
        'INITIAL_TIMESTAMP':1503430225,
        'INITIAL_HEIGHT':481824}

def fast_forward_selector(forward_blocks, total_blocks, FTL):
  def selector(states, n):
    total_jump = total_blocks*mining.IDEAL_BLOCK_TIME + FTL
    if n < forward_blocks:
      return states[-1].timestamp + total_jump // forward_blocks
    else:
      return states[-1].timestamp + 1
  return selector

def two_week_jump_selector(forward_blocks, total_blocks, FTL):
  def selector(states, n):
    #total_jump = 14 * 24 * 3600*1
    total_jump = total_blocks*mining.IDEAL_BLOCK_TIME + FTL
    if n % 2:
      return states[-1].timestamp - (total_jump - mining.IDEAL_BLOCK_TIME*n)
    else:
      return states[-1].timestamp + (total_jump - mining.IDEAL_BLOCK_TIME*n) + mining.IDEAL_BLOCK_TIME
  return selector


def simulate_attack(algoname, selector_type, forward_blocks=1, total_blocks=500, FTL=7200, autoend=False):
  algo = mining.Algos[algoname]

  total_jump = total_blocks*mining.IDEAL_BLOCK_TIME + FTL
  selector = selector_type(forward_blocks, total_blocks, FTL)
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
    timestamp = selector(states, n)
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
  alg, f, t, FTL = 'cw-144', 1, 145, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, True)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))

  alg, f, t, FTL = 'asert-144', 1, 283, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))

  alg, f, t, FTL = 'asert-576', 1, 1290, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, True)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))
  
  alg, f, t, FTL = 'wtema-144', 48, 330, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, True)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))
  alg, f, t, FTL = 'wtema-576', 75, 1200, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, True)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))

  alg, f, t, FTL = 'grasberg-neutral-288', 1, 804, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, False)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))
  alg, f, t, FTL = 'grasberg-nodrift-288', 1, 804, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, False)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))
  alg, f, t, FTL = 'aserti3-288', 1, 804, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, False)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))

   # optimum attack not yet found, but here's something kinda close
  alg, f, t, FTL = 'lwma-144', 1, 123, 7200
  s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, True)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))



  print()

  alg, f, t, FTL = 'grasberg-nodrift-288', 1, 804, 7200
  for t in range(400, 1200, 100):
    s, n, p, d = simulate_attack(alg, fast_forward_selector, f, t, FTL, False)
    print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t" % (alg, f, t, FTL, n, d*100, p))

  print()
  alg, f, t, FTL = 'grasberg-neutral-288', 1, 804, 7200
  for t in range(400, 1200, 100):
    s, n, p, d = simulate_attack(alg, two_week_jump_selector, f, t, FTL, False)
    print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t (two-week jumper)" % (alg, f, t, FTL, n, d*100, p))
  alg, f, t, FTL = 'grasberg-nodrift-288', 1, 804, 7200
  for t in range(400, 1200, 100):
    s, n, p, d = simulate_attack(alg, two_week_jump_selector, f, t, FTL, False)
    print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t (two-week jumper)" % (alg, f, t, FTL, n, d*100, p))

  print()
  alg, f, t, FTL = 'grasberg-neutral-288', 1, 800, 7200
  s, n, p, d = simulate_attack(alg, two_week_jump_selector, f, t, FTL, True)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t (two-week jumper)" % (alg, f, t, FTL, n, d*100, p))
  alg, f, t, FTL = 'grasberg-nodrift-288', 1, 800, 7200
  s, n, p, d = simulate_attack(alg, two_week_jump_selector, f, t, FTL, True)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t (two-week jumper)" % (alg, f, t, FTL, n, d*100, p))
  alg, f, t, FTL = 'aserti3-288', 1, 800, 7200
  s, n, p, d = simulate_attack(alg, two_week_jump_selector, f, t, FTL, True)
  print("Alg: %-20s fwd=%-3i tot=%-4i FTL=%i\tBlocks mined: %i\tEnd diff: %4.2f%%\tProfit: %4.2f%%\t (two-week jumper)" % (alg, f, t, FTL, n, d*100, p))
