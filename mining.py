#!/usr/bin/env python3

import argparse
import datetime
import math
import random
import statistics
import sys
import time
from collections import namedtuple
from functools import partial
from operator import attrgetter
from threading import Lock

def bitz_to_target(bitz):
    return bitz

def bits_to_bitz(bits):
    return bits_to_target(bits)

def bits_to_target(bits):
    size = bits >> 24
    assert size <= 0x1d

    word = bits & 0x00ffffff
    assert 0x8000 <= word <= 0x7fffff

    if size <= 3:
        return word >> (8 * (3 - size))
    else:
        return word << (8 * (size - 3))

MAX_BITS = 0x1d00ffff
MAX_BITZ = bits_to_bitz(MAX_BITS)
MAX_TARGET = bits_to_target(MAX_BITS)

def target_to_bitz(target):
    assert target > 0
    if target > MAX_TARGET:
        print('Warning: target went above maximum ({} > {})'
              .format(target, MAX_TARGET), file=sys.stderr)
        target = MAX_TARGET
    return target

def target_to_bits(target):
    assert target > 0
    if target > MAX_TARGET:
        print('Warning: target went above maximum ({} > {})'
              .format(target, MAX_TARGET), file=sys.stderr)
        target = MAX_TARGET
    size = (target.bit_length() + 7) // 8
    mask64 = 0xffffffffffffffff
    if size <= 3:
        compact = (target & mask64) << (8 * (3 - size))
    else:
        compact = (target >> (8 * (size - 3))) & mask64

    if compact & 0x00800000:
        compact >>= 8
        size += 1

    assert compact == (compact & 0x007fffff)
    assert size < 256
    return compact | size << 24

def bits_to_work(bits):
    return (2 << 255) // (bits_to_target(bits) + 1)

def bitz_to_work(bitz):
    return (2 << 255) // (bitz_to_target(bitz) + 1)

def target_to_hex(target):
    h = hex(target)[2:]
    return '0' * (64 - len(h)) + h

TARGET_1 = bits_to_target(486604799)

default_params = {
    'INITIAL_BCC_BITS':0x18084bb7,
    'INITIAL_SWC_BITS':0x18013ce9,
    'INITIAL_FX':0.19,
    'INITIAL_TIMESTAMP':1595564499,
    'INITIAL_HASHRATE':1000,    # In PH/s.
    'INITIAL_HEIGHT':645264,
    'BTC_fees':0.02,
    'BCH_fees':0.002,
    'num_blocks':10000,

    # Steady hashrate mines the BCC chain all the time.  In PH/s.
    'STEADY_HASHRATE':300,

    # Variable hash is split across both chains according to relative
    # revenue.  If the revenue ratio for either chain is at least 15%
    # higher, everything switches.  Otherwise the proportion mining the
    # chain is linear between +- 15%.
    'VARIABLE_HASHRATE':2000,   # In PH/s.
    'VARIABLE_PCT':15,   # 85% to 115%
    'VARIABLE_WINDOW':6,  # No of blocks averaged to determine revenue ratio
    'MEMORY_GAIN':.01,            # if rev_ratio is 1.01, then the next block's HR will be 0.01*MEMORY_GAIN higher

    # Greedy hashrate switches chain if that chain is more profitable for
    # GREEDY_WINDOW BCC blocks.  It will only bother to switch if it has
    # consistently been GREEDY_PCT more profitable.
    'GREEDY_HASHRATE':2000,     # In PH/s.
    'GREEDY_PCT':10,
    'GREEDY_WINDOW':6,
}
IDEAL_BLOCK_TIME = 10 * 60

State = namedtuple('State', 'height wall_time timestamp bitz chainwork fx '
                   'hashrate rev_ratio var_frac memory_frac greedy_frac msg')

states = []
lock = Lock() # hack to deal with concurrency and global use of states


def print_headers():
    print(', '.join(['Height', 'FX', 'Block Time', 'Unix', 'Timestamp',
                     'Difficulty (bn)', 'Implied Difficulty (bn)',
                     'Hashrate (PH/s)', 'Rev Ratio', 'memory_hashrate', 'Greedy?', 'Comments']))

def print_state():
    state = states[-1]
    block_time = state.timestamp - states[-2].timestamp
    t = datetime.datetime.fromtimestamp(state.timestamp)
    difficulty = TARGET_1 / bitz_to_target(state.bitz)
    implied_diff = TARGET_1 / ((2 << 255) / (state.hashrate * 1e15 * IDEAL_BLOCK_TIME))
    print(', '.join(['{:d}'.format(state.height),
                     '{:.8f}'.format(state.fx),
                     '{:d}'.format(block_time),
                     '{:d}'.format(state.timestamp),
                     '{:%Y-%m-%d %H:%M:%S}'.format(t),
                     '{:.2f}'.format(difficulty / 1e9),
                     '{:.2f}'.format(implied_diff / 1e9),
                     '{:.0f}'.format(state.hashrate),
                     '{:.3f}'.format(state.rev_ratio),
                     'Yes' if state.greedy_frac == 1.0 else 'No',
                     state.msg]))

def revenue_ratio(fx, BCC_target, params):
    '''Returns the instantaneous SWC revenue rate divided by the
    instantaneous BCC revenue rate.  A value less than 1.0 makes it
    attractive to mine BCC.  Greater than 1.0, SWC.'''
    SWC_fees = params['BTC_fees'] * random.random()
    SWC_revenue = 12.5 + SWC_fees
    SWC_target = bits_to_target(default_params['INITIAL_SWC_BITS'])

    BCC_fees = params['BCH_fees'] * random.random()
    BCC_revenue = (12.5 + BCC_fees) * fx

    SWC_difficulty_ratio = BCC_target / SWC_target
    return SWC_revenue / SWC_difficulty_ratio / BCC_revenue

def median_time_past(states):
    times = [state.timestamp for state in states]
    return sorted(times)[len(times) // 2]

def suitable_block_index(index):
    assert index >= 2
    indices = [index - 2, index - 1, index]

    if states[indices[0]].timestamp > states[indices[2]].timestamp:
        indices[0], indices[2] = indices[2], indices[0]

    if states[indices[0]].timestamp > states[indices[1]].timestamp:
        indices[0], indices[1] = indices[1], indices[0]

    if states[indices[1]].timestamp > states[indices[2]].timestamp:
        indices[1], indices[2] = indices[2], indices[1]

    return indices[1]

def compute_index_fast(index_last):
    for candidate in range(index_last - 3, 0, -1):
        index_fast = suitable_block_index(candidate)
        if index_last - index_fast < 5:
            continue
        if (states[index_last].timestamp - states[index_fast].timestamp
            >= 13 * IDEAL_BLOCK_TIME):
            return index_fast
    raise AssertionError('should not happen')

def compute_target(first_index, last_index):
    work = states[last_index].chainwork - states[first_index].chainwork
    work *= IDEAL_BLOCK_TIME
    work //= states[last_index].timestamp - states[first_index].timestamp
    return (2 << 255) // work - 1

def compute_cw_target(block_count):
    N = len(states) - 1
    last = suitable_block_index(N)
    first = suitable_block_index(N - block_count)
    timespan = states[last].timestamp - states[first].timestamp
    timespan = max(block_count * IDEAL_BLOCK_TIME // 2, min(block_count * 2 * IDEAL_BLOCK_TIME, timespan))
    work = (states[last].chainwork - states[first].chainwork) * IDEAL_BLOCK_TIME // timespan
    return (2 << 255) // work - 1

def next_bitz_wtema(msg, alpha_recip):
    # This algorithm is weighted-target exponential moving average.
    # Target is calculated based on inter-block times weighted by a
    # progressively decreasing factor for past inter-block times,
    # according to the parameter alpha.  If the single_block_target SBT is
    # calculated as:
    #    SBT = prior_target * block_time / ideal_block_time
    # then:
    #    next_target = SBT * α + prior_target * (1 - α)
    # Substituting and factorizing:
    #    next_target = prior_target * α / ideal_block_time
    #                  * (block_time + (1 / α - 1) * ideal_block_time)
    # We use the reciprocal of alpha as an integer to avoid floating
    # point arithmetic.  Doing so the above formula maintains precision and
    # avoids overflows wih large targets in regtest
    #if states[-1].height % 2000 == 0:
    #    return next_bits_cw(msg, 2000)

    prior_target = bitz_to_target(states[-1].bitz)
    block_time = max(0, states[-1].timestamp - states[-2].timestamp) # avoid negative solvetimes
    next_target = prior_target // (IDEAL_BLOCK_TIME * alpha_recip)
    next_target *= block_time + IDEAL_BLOCK_TIME * (alpha_recip - 1)
    # Constrain individual target changes to 50%
    max_change = prior_target >> 1
    assert next_target >= prior_target - max_change
    assert next_target <= prior_target + max_change
    # next_target = max(min(next_target, prior_target + max_change), prior_target - max_change)
    return target_to_bitz(next_target)

def next_bitz_grin(msg, n, dampen):
    delta_ts       = max(0, states[-1].timestamp - states[-1-n].timestamp) # avoid negative solvetimes
    delta_work     = states[-1].chainwork - states[-1-n].chainwork

    damped_ts = (1 * delta_ts + (dampen - 1) * (n * IDEAL_BLOCK_TIME) ) // dampen
    work = delta_work * IDEAL_BLOCK_TIME // damped_ts
    return target_to_bitz((2 << 255) // work - 1)

def block_time(mean_time, **params):
    if 'deterministic' in params:
        return mean_time

    k = params['bobtail'] if 'bobtail' in params else 1
    if k==1:
        # gammavariate treats random seeds slightly differently from expovariate or random.random(),
        # so we use expovariate when continuity of test results might matter
        return random.expovariate(1/mean_time)
    return mean_time*random.gammavariate(k, 1/k)


def next_fx_random(r, **params):
    return states[-1].fx * (1.0 + (r - 0.5) / 200)

def next_fx_constant(r, **params):
    return states[-1].fx

def next_fx_ramp(r, **params):
    return states[-1].fx * 1.00017149454

def next_hashrate(states, scenario, params):
    msg = []
    high = 1.0 + params['VARIABLE_PCT'] / 100
    scale_fac = 50 / params['VARIABLE_PCT']
    N = params['VARIABLE_WINDOW']
    mean_rev_ratio = sum(state.rev_ratio for state in states[-N:]) / N

    if 1:
        var_fraction = (high - mean_rev_ratio**params['VARIABLE_EXPONENT']) * scale_fac
        memory_frac = states[-1].memory_frac +  ((var_fraction-.5) * params['MEMORY_GAIN'])
        var_fraction = max(0, min(1, var_fraction + memory_frac))
    else:
        var_fraction = (high - mean_rev_ratio**params['VARIABLE_EXPONENT']) * scale_fac
        memory_frac  = states[-1].memory_frac * 2**(1*(1-mean_rev_ratio))
        if var_fraction > 0 and memory_frac <= 0:
            memory_frac = var_fraction
        var_fraction = ((1-mean_rev_ratio**params['VARIABLE_EXPONENT'])*4 + states[-1].memory_frac)
        a = 0.2
        memory_frac = (1-a) * states[-1].memory_frac + a * var_fraction
        var_fraction = 2**(var_fraction*20)
        memory_frac = max(-20, min(0, memory_frac))
        var_fraction = max(0, min(1, var_fraction))
    

    if ((scenario.pump_144_threshold > 0) and
        (states[-1-144+5].timestamp - states[-1-144].timestamp > scenario.pump_144_threshold)):
        var_fraction = max(var_fraction, .25)


    # mem_rev_ratio = sum(state.rev_ratio for state in states[-params['MEMORY_WINDOW']:]) / params['MEMORY_WINDOW']
    # memdelta = (1-mem_rev_ratio**params['MEMORY_POWER'])*params['MEMORY_GAIN']
    # if params['MEMORY_REMAINING']:
    #     memory_frac = states[-1].memory_frac + memdelta*(1-states[-1].memory_frac)
    # else:
    #     memory_frac = states[-1].memory_frac + memdelta
    # memory_frac = max(0.0, min(1.0, memory_frac))

    N = params['GREEDY_WINDOW']
    gready_rev_ratio = sum(state.rev_ratio for state in states[-N:]) / N
    greedy_frac = states[-1].greedy_frac
    if mean_rev_ratio >= 1 + params['GREEDY_PCT'] / 100:
        if greedy_frac != 0.0:
            msg.append("Greedy miners left")
        greedy_frac = 0.0
    elif mean_rev_ratio <= 1 - params['GREEDY_PCT'] / 100:
        if greedy_frac != 1.0:
            msg.append("Greedy miners joined")
        greedy_frac = 1.0

    hashrate = (params['STEADY_HASHRATE'] + scenario.dr_hashrate
                + params['VARIABLE_HASHRATE'] * var_fraction
                #+ params['MEMORY_HASHRATE'] * memory_frac
                + params['GREEDY_HASHRATE'] * greedy_frac)

    return hashrate, msg, var_fraction, memory_frac, greedy_frac

def next_step(fx_jump_factor, params):
    algo, scenario = params['algo'], params['scenario']
    hashrate, msg, var_frac, memory_frac, greedy_frac = next_hashrate(states, scenario, params)
    # First figure out our hashrate
    # Calculate our dynamic difficulty
    bitz = algo.next_bitz(msg, **algo.params)
    target = bitz_to_target(bitz)
    # See how long we take to mine a block
    mean_hashes = pow(2, 256) // target
    mean_time = mean_hashes / (hashrate * 1e15)
    time = int(block_time(mean_time, **scenario.params) + 0.5)
    wall_time = states[-1].wall_time + time
    # Did the difficulty ramp hashrate get the block?
    if random.random() < (abs(scenario.dr_hashrate) / hashrate):
        if (scenario.dr_hashrate > 0):
            timestamp = median_time_past(states[-11:]) + 1
        else:
            timestamp = wall_time + 2 * 60 * 60
    else:
        timestamp = wall_time
    # Get a new FX rate
    rand = random.random()
    fx = scenario.next_fx(rand, **scenario.params)
    if fx_jump_factor != 1.0:
        msg.append('FX jumped by factor {:.2f}'.format(fx_jump_factor))
        fx *= fx_jump_factor
    rev_ratio = revenue_ratio(fx, target, params)

    chainwork = states[-1].chainwork + bitz_to_work(bitz)

    # add a state
    states.append(State(states[-1].height + 1, wall_time, timestamp,
                        bitz, chainwork, fx, hashrate, rev_ratio,
                        var_frac, memory_frac, greedy_frac, ' / '.join(msg)))

Algo = namedtuple('Algo', 'next_bitz params')

Algos = {
    'grin-60-3' : Algo(next_bitz_grin, {
        'n': 60,
        'dampen': 3,
    }),
    'grin-60-5' : Algo(next_bitz_grin, {
        'n': 60,
        'dampen': 5,
    }),
    'grin-60-7' : Algo(next_bitz_grin, {
        'n': 60,
        'dampen': 7,
    }),
    'wtema-120' : Algo(next_bitz_wtema, {
        'alpha_recip': 120, 
    }),
    'wtema-180' : Algo(next_bitz_wtema, {
        'alpha_recip': 180,
    }),
    'wtema-240' : Algo(next_bitz_wtema, {
        'alpha_recip': 240,
    }),
}

Scenario = namedtuple('Scenario', 'next_fx, params, dr_hashrate, pump_144_threshold')

Scenarios = {
    'default'      : Scenario(next_fx_random, {}, 0, 0),
    'bobtail10'    : Scenario(next_fx_random,   {"bobtail":10},  0, 0),
    'bobtail100'   : Scenario(next_fx_random,   {"bobtail":100}, 0, 0),
    'deterministic': Scenario(next_fx_random,   {"deterministic":True}, 0, 0),
    'stable'       : Scenario(next_fx_constant, {"price1x":True}, 0, 0),
    'fxramp'       : Scenario(next_fx_ramp, {}, 0, 0),
    # Difficulty rampers with given PH/s
    'dr50'         : Scenario(next_fx_random, {}, 50, 0),
    'dr75'         : Scenario(next_fx_random, {}, 75, 0),
    'dr100'        : Scenario(next_fx_random, {}, 100, 0),
    'pump-osc'     : Scenario(next_fx_ramp,   {}, 0, 8000),
    'ft50'         : Scenario(next_fx_random, {}, -50, 0),
    'ft100'        : Scenario(next_fx_random, {}, -100, 0),
    'price10x'     : Scenario(next_fx_random, {"price10x":True}, 0, 0),
}

def run_one_simul(print_it, returnstate=False, params=default_params):
    lock.acquire()
    states.clear()

    try:
        # Initial state is afer 2020 steady prefix blocks
        N = 2020
        for n in range(-N, 0):
            state = State(params['INITIAL_HEIGHT'] + n, params['INITIAL_TIMESTAMP'] + n * IDEAL_BLOCK_TIME,
                          params['INITIAL_TIMESTAMP'] + n * IDEAL_BLOCK_TIME,
                          bits_to_bitz(params['INITIAL_BCC_BITS']), bits_to_work(params['INITIAL_BCC_BITS']) * (n + N + 1),
                          params['INITIAL_FX'], params['INITIAL_HASHRATE'], 0.0, 0.5, 0.0, False, '')
            states.append(state)

        # Add a few randomly-timed FX jumps (up or down 10 and 15 percent) to
        # see how algos recalibrate
        fx_jumps = {}
        num_fx_jumps = 2 + params['num_blocks']/3000

        if 'price10x' in params['scenario'].params:
            factor_choices = [0.1, 0.25, 0.5, 2.0, 4.0, 10.0]
            for n in range(4):
                fx_jumps[random.randrange(params['num_blocks'])] = random.choice(factor_choices)
        elif not 'price1x' in params['scenario'].params:
            factor_choices = [0.85, 0.9, 1.1, 1.15]
            for n in range(10):
                fx_jumps[random.randrange(params['num_blocks'])] = random.choice(factor_choices)

        # Run the simulation
        if print_it:
            print_headers()
        for n in range(params['num_blocks']):
            fx_jump_factor = fx_jumps.get(n, 1.0)
            next_step(fx_jump_factor, params)
            if print_it:
                print_state()

        # Drop the prefix blocks to be left with the simulation blocks
        simul = states[N:]
    except:
        lock.release()
        raise
    lock.release()

    if returnstate:
        return simul

    block_times = [simul[n + 1].timestamp - simul[n].timestamp
                   for n in range(len(simul) - 1)]
    return block_times


def main():
    '''Outputs CSV data to stdout.   Final stats to stderr.'''

    parser = argparse.ArgumentParser('Run a mining simulation')
    parser.add_argument('-a', '--algo', metavar='algo', type=str,
                        choices = list(Algos.keys()),
                        default = 'k-1', help='algorithm choice')
    parser.add_argument('-s', '--scenario', metavar='scenario', type=str,
                        choices = list(Scenarios.keys()),
                        default = 'default', help='scenario choice')
    parser.add_argument('-r', '--seed', metavar='seed', type=int,
                        default = None, help='random seed')
    parser.add_argument('-n', '--count', metavar='count', type=int,
                        default = 1, help='count of simuls to run')
    args = parser.parse_args()

    count = max(1, args.count)
    algo = Algos.get(args.algo)
    scenario = Scenarios.get(args.scenario)
    seed = int(time.time()) if args.seed is None else args.seed

    to_stderr = partial(print, file=sys.stderr)
    to_stderr("Starting seed {} for {} simuls".format(seed, count))

    means = []
    std_devs = []
    medians = []
    maxs = []
    for loop in range(count):
        random.seed(seed)
        seed += 1
        block_times = run_one_simul(algo, scenario, count == 1)
        means.append(statistics.mean(block_times))
        std_devs.append(statistics.stdev(block_times))
        medians.append(sorted(block_times)[len(block_times) // 2])
        maxs.append(max(block_times))

    def stats(text, values):
        if count == 1:
            to_stderr('{} {}s'.format(text, values[0]))
        else:
            to_stderr('{}(s) Range {:0.1f}-{:0.1f} Mean {:0.1f} '
                      'Std Dev {:0.1f} Median {:0.1f}'
                      .format(text, min(values), max(values),
                              statistics.mean(values),
                              statistics.stdev(values),
                              sorted(values)[len(values) // 2]))

    stats("Mean   block time", means)
    stats("StdDev block time", std_devs)
    stats("Median block time", medians)
    stats("Max    block time", maxs)

if __name__ == '__main__':
    main()
