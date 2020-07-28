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
MAX_TARGET = bits_to_target(MAX_BITS)

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

State = namedtuple('State', 'height wall_time timestamp bits chainwork fx '
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
    difficulty = TARGET_1 / bits_to_target(state.bits)
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

def next_bits_k(msg, mtp_window, high_barrier, target_raise_frac,
                low_barrier, target_drop_frac, fast_blocks_pct):
    # Calculate N-block MTP diff
    MTP_0 = median_time_past(states[-11:])
    MTP_N = median_time_past(states[-11-mtp_window:-mtp_window])
    MTP_diff = MTP_0 - MTP_N
    bits = states[-1].bits
    target = bits_to_target(bits)

    # Long term block production time stabiliser
    t = states[-1].timestamp - states[-2017].timestamp
    if t < IDEAL_BLOCK_TIME * 2016 * fast_blocks_pct // 100:
        msg.append("2016 block time difficulty raise")
        target -= target // target_drop_frac

    if MTP_diff > high_barrier:
        target += target // target_raise_frac
        msg.append("Difficulty drop {}".format(MTP_diff))
    elif MTP_diff < low_barrier:
        target -= target // target_drop_frac
        msg.append("Difficulty raise {}".format(MTP_diff))
    else:
        msg.append("Difficulty held {}".format(MTP_diff))

    return target_to_bits(target)

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

def next_bits_d(msg):
    N = len(states) - 1
    index_last = suitable_block_index(N)
    index_first = suitable_block_index(N - 2016)
    interval_target = compute_target(index_first, index_last)
    index_fast = compute_index_fast(index_last)
    fast_target = compute_target(index_fast, index_last)

    next_target = interval_target
    if (fast_target < interval_target - (interval_target >> 2) or
        fast_target > interval_target + (interval_target >> 2)):
        msg.append("fast target")
        next_target = fast_target
    else:
        msg.append("interval target")

    prev_target = bits_to_target(states[-1].bits)
    min_target = prev_target - (prev_target >> 3)
    if next_target < min_target:
        msg.append("min target")
        return target_to_bits(min_target)

    max_target = prev_target + (prev_target >> 3)
    if next_target > max_target:
        msg.append("max target")
        return target_to_bits(max_target)

    return target_to_bits(next_target)

def compute_cw_target(block_count):
    N = len(states) - 1
    last = suitable_block_index(N)
    first = suitable_block_index(N - block_count)
    timespan = states[last].timestamp - states[first].timestamp
    timespan = max(block_count * IDEAL_BLOCK_TIME // 2, min(block_count * 2 * IDEAL_BLOCK_TIME, timespan))
    work = (states[last].chainwork - states[first].chainwork) * IDEAL_BLOCK_TIME // timespan
    return (2 << 255) // work - 1

def next_bits_sha(msg):
    primes = [73, 79, 83, 89, 97,
              101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151]

    # The timestamp % len(primes) is a proxy for previous
    # block SHAx2 % len(primes), but that data is not available
    # in this simulation
    prime = primes[states[-1].timestamp % len(primes)]

    interval_target = compute_cw_target(prime)
    return target_to_bits(interval_target)

def next_bits_cw(msg, block_count):
    interval_target = compute_cw_target(block_count)
    return target_to_bits(interval_target)

def next_bits_wt(msg, block_count):
    first, last  = -1-block_count, -1
    timespan = 0
    prior_timestamp = states[first].timestamp
    for i in range(first + 1, last + 1):
        target_i = bits_to_target(states[i].bits)

        # Prevent negative time_i values
        timestamp = max(states[i].timestamp, prior_timestamp)
        time_i = timestamp - prior_timestamp
        prior_timestamp = timestamp
        adj_time_i = time_i * target_i # Difficulty weight
        timespan += adj_time_i * (i - first) # Recency weight

    timespan = timespan * 2 // (block_count + 1) # Normalize recency weight
    target = timespan // (IDEAL_BLOCK_TIME * block_count)
    return target_to_bits(target)

def next_bits_wt_compare(msg, block_count):
    with open("current_state.csv", 'w') as fh:
        for s in states:
            fh.write("%s,%s,%s\n" % (s.height, s.bits, s.timestamp))

    from subprocess import Popen, PIPE

    process = Popen(["./cashwork"], stdout=PIPE)
    (next_bits, err) = process.communicate()
    exit_code = process.wait()

    next_bits = int(next_bits.decode())
    next_bits_py = next_bits_wt(msg, block_count)
    if next_bits != next_bits_py:
        print("ERROR: Bits don't match. External %s, local %s" % (next_bits, next_bits_py))
        assert(next_bits == next_bits_py)
    return next_bits

def next_bits_wtema(msg, alpha_recip, mo3=0):
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

    if mo3:
        last = suitable_block_index(len(states) - 1)
        first = suitable_block_index(len(states) - 2)
    else:
        last = -1
        first = -2
    if mo3==2:
        # This version results in nasty 2-block oscillations
        prior_target = bits_to_target(states[last].bits)
    else:
        prior_target = bits_to_target(states[-1].bits)

    block_time = states[last].timestamp - states[first].timestamp
    next_target = prior_target // (IDEAL_BLOCK_TIME * alpha_recip)
    next_target *= block_time + IDEAL_BLOCK_TIME * (alpha_recip - 1)
    return target_to_bits(next_target)

def next_bits_asert(msg, tau):
    blocks_time = states[-1].timestamp - states[0].timestamp
    height_diff = states[-1].height - states[0].height
    orig_target = bits_to_target(states[-0].bits)
    next_target = int(orig_target * math.e**((blocks_time - IDEAL_BLOCK_TIME*height_diff) / tau))
    return target_to_bits(next_target)

def next_bits_asert_discrete(msg, window, granularity):
    """Another exponential-decay-based algo that uses integer math instead of exponentiation.  As with asert, we increase difficulty by a fixed amount each time a block is found,
    and decrease it steadily for the passage of time between found blocks.  But here both adjustments are done in integer math, using the principles that:
    1. We can discretize "Decrease difficulty steadily by a factor of 1/e over (say) 1 day", into "Decrease difficulty by exactly 1/e^(1/100) for each 1/100 of a day."
    2. We can closely approximate "difficulty * 1/e^(1/100)", by "(difficulty * 4252231657) >> 32".  (Really, just "difficulty * 99 // 100" would probably do the job too.)

    The "window" param is meant to invoke the fixed time window of a simple moving average, but here we give it the standard equivalent EMA interpretation: the window is "how old a
    block has to be (in seconds) for us to discount its weight by a factor of e."  So, instead of 86400 (1 day) meaning "average the block times over the last day," here it means 
    "in our weighted avg of block times, give a day-old block 1/e the weight of the latest block."  This results in comparable responsiveness to a fixed 1-day window.

    Given this framework, we update the target as follows:
    1. Decrease the difficulty (ie, increase the target) by the constant factor we always increase it by for each new block.  (See the ASERT algo for an explanation of this.)
    2. Adjust the difficulty based on the passage of time (typically increase it, except in the unusual case where this block's timestamp is before the previous block's):
       a) Specify a granularity - the number of segments we'll divide the time window into.  Eg, window = 86400 and granularity = 100, means segment = 864.
       b) Figure out which numbered segment (since genesis) the previous block's timestamp fell into, and which segment the current block falls into.
       c) The difference between the segment numbers of the two blocks tells us how many discrete "per-segment difficulty adjustments" we need to make.  Eg, if the current block
          lands in the segment 3 segments after the one the previous block did, we need to decrease difficulty by three "segment adjustments".
       d) Theoretically, if granularity = 100, we should multiply difficulty by e**(-1/100) for each segment.  We can approximate each adjustment by an int-math multiplication, 
          using the e^(-1/100)*(2**32) in TIME_SEGMENT_DIFFICULTY_DECREASE_FACTOR above.  This means our algo can only time-adjust difficulty by multiples of 1% - but that's OK."""

    FACTOR_SCALING_BITS = 32
    # These factors are scaled by 2**FACTOR_SCALING_BITS.  Eg, (x * 4252231657) >> 32, is approximately x * e^(-1/100).
    TIME_SEGMENT_DIFFICULTY_DECREASE_FACTOR = {
        100: 4252231657,                    # If current block's timestamp is one segment later than previous block's, then multiply difficulty by this number and divide by 2**32.
    }
    BLOCK_DIFFICULTY_INCREASE_FACTOR = {
        144: 4324897261,                    # If window = 144 * IDEAL_BLOCK_TIME, then every time a block is found, multiply difficulty by this number and divide by 2**32.
    }

    old_segment_number = states[-2].timestamp // (window // granularity)
    new_segment_number = states[-1].timestamp // (window // granularity)
    old_target = bits_to_target(states[-1].bits)

    # We divide by the factors here, rather than multiply, because we're actually adjusting target, not difficulty:
    new_target = (old_target << FACTOR_SCALING_BITS) // BLOCK_DIFFICULTY_INCREASE_FACTOR[window // IDEAL_BLOCK_TIME]
    if new_segment_number > old_segment_number:
        # Doing this in a simple for loop means that a pathological block time (far in past or future) could make this very slow.  I'm not sure such blocks ever actually occur,
        # but if this were a concern we could speed this up via https://en.wikipedia.org/wiki/Exponentiation_by_squaring.  Eg, if the two block times are 20 segments apart, a naive
        # loop does 20 multiplications/divisions (plus bit-shifts), whereas exp by squaring could do it in 6 as follows:
        #     e**(2/100)  = (e**(1/100))**2
        #     e**(4/100)  = (e**(2/100))**2
        #     e**(8/100)  = (e**(4/100))**2
        #     e**(16/100) = (e**(8/100))**2
        #     e**(20/100) = e**(16/100) * e**(4/100)
        #     new_target = old_target * e**(20/100)
        # This saving gets significant for large numbers (log vs linear): if the blocks were 1,000,000 segments apart, this would mean 26 multiplications rather than 1,000,000.
        for _ in range(old_segment_number, new_segment_number):
            new_target = (new_target << FACTOR_SCALING_BITS) // TIME_SEGMENT_DIFFICULTY_DECREASE_FACTOR[granularity]
    elif new_segment_number < old_segment_number:                       # If the new block's timestamp is weirdly before the old one's, OK then, *increase* difficulty accordingly
        for _ in range(new_segment_number, old_segment_number):
            new_target = (new_target * TIME_SEGMENT_DIFFICULTY_DECREASE_FACTOR[granularity]) >> FACTOR_SCALING_BITS

    return target_to_bits(new_target)

def next_bits_aserti(msg, tau, mode=1, mo3=False):
    rbits = 16      # number of bits after the radix for fixed-point math
    radix = 1<<rbits
    if mo3:
        last = suitable_block_index(len(states) - 1)
        first = suitable_block_index(2)
    else:
        last = len(states)-1
        first = 0

    blocks_time = states[last].timestamp - states[first].timestamp
    height_diff = states[last].height    - states[first].height
    target = bits_to_target(states[-0].bits)

    # Ultimately, we want to approximate the following ASERT formula, using only integer (fixed-point) math:
    #     new_target = old_target * 2^((blocks_time - IDEAL_BLOCK_TIME*(height_diff+1)) / tau)

    # First, we'll calculate the exponent, using floor division:
    exponent = int(((blocks_time - IDEAL_BLOCK_TIME*height_diff) * radix) / tau)

    # Next, we use the 2^x = 2 * 2^(x-1) identity to shift our exponent into the (0, 1] interval.
    # First, the truncated exponent tells us how many shifts we need to do
    shifts = exponent >> rbits

    # Next, we shift. Python doesn't allow shifting by negative integers, so:
    if shifts < 0:
        target >>= -shifts
    else:
        target <<= shifts
    exponent -= shifts*radix

    # Now we compute an approximated target * 2^(exponent)
    if mode == 1:
        # target * 2^x ~= target * (1 + x)
        target += (target * exponent) >> rbits
    elif mode == 2:
        # target * 2^x ~= target * (1 + 2*x/3 + x**2/3)
        target += (target * 2*exponent*radix//3 + target*exponent*exponent //3) >> (rbits*2)
    elif mode == 3:
        # target * 2^x ~= target * (1 + 0.695502049*x + 0.2262698*x**2 + 0.0782318*x**3)
        factor = (195766423245049*exponent + 971821376*exponent**2 + 5127*exponent**3 + 2**47)>>(rbits*3)
        target += (target * factor) >> rbits
    return target_to_bits(target)

def next_bits_grasberg(msg, nblocks, genesis_time=1231006505, correct_drift=True):
    LN2_32 = 2977044472
    POW2_32 = 1 << 32
    def deterministicExp2(n):
        """
        Rescale the computation depending on n for better precision.
        We use the MSB to form 16 buckets.
        """
        if n < 0: n += 1<<32
        bucket = n >> 28;

          # Rescale around the middle of the range via:
          #     exp2(n) = 2^32 * 2^(n/2^32)
          #             = 2^32 * 2^((n - d)/2^32 + d/2^32)
          #             = 2^32 * 2^(d/2^32) * 2^((n - d)/2^32)
          # Using x = n - d:
          #     exp2(n) = 2^32 * 2^(d/2^32) * 2^(x/2^32)

        d = (2 * bucket + 1) << 27;
        x = n - d;

        k0s = [
            # 2^32 * (2^(1/32) - 1)  = 94047537.3451
            94047537,
            # 2^32 * (2^(3/32) - 1)  = 288365825.147
            288365825,
            # 2^32 * (2^(5/32) - 1)  = 491287318.545
            491287319,
            # 2^32 * (2^(7/32) - 1)  = 703192913.992
            703192914,
            # 2^32 * (2^(9/32) - 1)  = 924480371.666
            924480372,
            # 2^32 * (2^(11/32) - 1) = 1155565062.10
            1155565062,
            # 2^32 * (2^(13/32) - 1) = 1396880745.83
            1396880746,
            # 2^32 * (2^(15/32) - 1) = 1648880387.65
            1648880388,
            # 2^32 * (2^(17/32) - 1) = 1912037006.77
            1912037007,
            # 2^32 * (2^(19/32) - 1) = 2186844564.80
            2186844565,
            # 2^32 * (2^(21/32) - 1) = 2473818892.86
            2473818893,
            # 2^32 * (2^(23/32) - 1) = 2773498659.88
            2773498660,
            # 2^32 * (2^(25/32) - 1) = 3086446383.71
            3086446384,
            # 2^32 * (2^(27/32) - 1) = 3413249486.97
            3413249487,
            # 2^32 * (2^(29/32) - 1) = 3754521399.73
            3754521400,
            # 2^32 * (2^(31/32) - 1) = 4110902710.89
            4110902711,
        ]
        k0 = k0s[bucket];

        k1s = [
            # 2^32 * ln(2) * 2^(1/32)  = 3042233257.17
            3042233257,
            # 2^32 * ln(2) * 2^(3/32)  = 3176924430.49
            3176924430,
            # 2^32 * ln(2) * 2^(5/32)  = 3317578891.51
            3317578892,
            # 2^32 * ln(2) * 2^(7/32)  = 3464460657.54
            3464460658,
            # 2^32 * ln(2) * 2^(9/32)  = 3617845434.92
            3617845435,
            # 2^32 * ln(2) * 2^(11/32) = 3778021136.56
            3778021137,
            # 2^32 * ln(2) * 2^(13/32) = 3945288422.37
            3945288422,
            # 2^32 * ln(2) * 2^(15/32) = 4119961263.60
            4119961264,
            # 2^32 * ln(2) * 2^(17/32) = 4302367532.19
            4302367532,
            # 2^32 * ln(2) * 2^(19/32) = 4492849616.23
            4492849616,
            # 2^32 * ln(2) * 2^(21/32) = 4691765062.62
            4691765063,
            # 2^32 * ln(2) * 2^(23/32) = 4899487248.21
            4899487248,
            # 2^32 * ln(2) * 2^(25/32) = 5116406080.64
            5116406081,
            # 2^32 * ln(2) * 2^(27/32) = 5342928730.26
            5342928730,
            # 2^32 * ln(2) * 2^(29/32) = 5579480394.39
            5579480394,
            # 2^32 * ln(2) * 2^(31/32) = 5826505095.43
            5826505095,
        ]
        k1 = k1s[bucket];

         # Now we aproximate the result using a taylor series.

        u0 = k0;
        u1_31 = (x * k1) >> 1;
        u2_31 = (((x * LN2_32) >> 32) * ((x * k1) >> 32)) >> 2;

        return u0 + ((u1_31 + u2_31) >> 31);

    def computeTargetBlockTime(states, genesis_time):
        lastBlockTime = states[-1].timestamp                        # apparently this one is the nTime, not the solvetime
        expectedTime = states[-1].height * IDEAL_BLOCK_TIME + genesis_time
        drift = expectedTime - lastBlockTime;
        tau = 14 * 24 * 60 * 60 # this is different from the ASERT tau, but shares the same name
        x32 = int((drift * POW2_32) / tau)
        # 2^32 * ln2(675/600) = 729822323.967
        X_CLIP = 729822324
        # We clip to ensure block time stay around 10 minutes in practice.
        x = max(min(x32, X_CLIP), -X_CLIP)
        offsetTime32 = IDEAL_BLOCK_TIME * deterministicExp2(x)
        return (IDEAL_BLOCK_TIME + (offsetTime32 >> 32)) >> (x32 < 0)

    def ComputeNextWork(states, nblocks, genesis_time, correct_drift):
        targetBlockTime = computeTargetBlockTime(states, genesis_time) if correct_drift else IDEAL_BLOCK_TIME
        lastBlockTime = states[-1].timestamp - states[-2].timestamp # apparently this one is the solvetime, not the nTime
        timeOffset = targetBlockTime - lastBlockTime
        tau32 = nblocks * IDEAL_BLOCK_TIME * LN2_32
        x32 = int((timeOffset * POW2_32) / (tau32 >> 32))
        xi = x32 >> 32
        xd = x32 & 0xffffffff
        lastBlockWork = int(states[-1].chainwork - states[-2].chainwork)
        if xi >= 32:
            return lastBlockWork << 32
        elif xi <= -32:
            return lastBlockWork >> 32
        offsetWork32 = lastBlockWork * deterministicExp2(xd)
        nextWork = (lastBlockWork + (offsetWork32 >> 32)) >> (-xi) if xi < 0 \
              else (lastBlockWork << xi) + (offsetWork32 >> (32 - xi))
        return int(nextWork)

    def ComputeTargetFromWork(work):
        # We need to compute T = (2^256 / W) - 1 but 2^256 doesn't fit in 256 bits.
        # By expressing 1 as W / W, we get (2^256 - W) / W, and we can compute
        # 2^256 - W as the complement of W.
        return (2**256-work)//work

    nextWork = ComputeNextWork(states, nblocks, genesis_time, correct_drift)
    nextTarget = ComputeTargetFromWork(nextWork)
    powLimit = 1<<224
    if nextTarget > powLimit: return target_to_bits(powLimit)
    return target_to_bits(nextTarget)

def next_bits_lwma(msg, n):
    block_intervals = [states[-(1+i)].timestamp - states[-(2+i)].timestamp for i in range(n)]
    block_works     = [states[-(1+i)].chainwork - states[-(2+i)].chainwork for i in range(n)]
    weighted_intervals = [block_intervals[i] * (n-i) for i in range(n)]
    weighted_works     = [block_works[i]     * (n-i) for i in range(n)]

    weighted_timespan = sum(weighted_intervals)
    weighted_work = sum(weighted_works)
    work = (weighted_work) * IDEAL_BLOCK_TIME // weighted_timespan
    return target_to_bits((2 << 255) // work - 1)

def next_bits_dgw3(msg, block_count):
    ''' Dark Gravity Wave v3 from Dash '''
    block_reading = -1 # dito
    counted_blocks = 0
    last_block_time = 0
    actual_time_span = 0
    past_difficulty_avg = 0
    past_difficulty_avg_prev = 0
    i = 1
    while states[block_reading].height > 0:
        if i > block_count:
            break
        counted_blocks += 1
        if counted_blocks <= block_count:
            if counted_blocks == 1:
                past_difficulty_avg = bits_to_target(states[block_reading].bits)
            else:
                past_difficulty_avg = ((past_difficulty_avg_prev * counted_blocks) + bits_to_target(states[block_reading].bits)) // ( counted_blocks + 1 )
        past_difficulty_avg_prev = past_difficulty_avg
        if last_block_time > 0:
            diff = last_block_time - states[block_reading].timestamp
            actual_time_span += diff
        last_block_time = states[block_reading].timestamp
        block_reading -= 1
        i += 1
    target_time_span = counted_blocks * IDEAL_BLOCK_TIME
    target = past_difficulty_avg
    if actual_time_span < (target_time_span // 3):
        actual_time_span = target_time_span // 3
    if actual_time_span > (target_time_span * 3):
        actual_time_span = target_time_span * 3
    target = target // target_time_span
    target *= actual_time_span
    if target > MAX_TARGET:
        return MAX_BITS
    else:
        return target_to_bits(int(target))

def next_bits_m2(msg, window_1, window_2):
    interval_target = compute_target(-1 - window_1, -1)
    interval_target += compute_target(-2 - window_2, -2)
    return target_to_bits(interval_target >> 1)

def next_bits_m4(msg, window_1, window_2, window_3, window_4):
    interval_target = compute_target(-1 - window_1, -1)
    interval_target += compute_target(-2 - window_2, -2)
    interval_target += compute_target(-3 - window_3, -3)
    interval_target += compute_target(-4 - window_4, -4)
    return target_to_bits(interval_target >> 2)

def next_bits_ema(msg, window):
    """This calculates difficulty (1/target) as proportional to the recent hashrate, where "recent hashrate" is estimated by an EMA (exponential moving avg) of recent "hashrate observations", and
    a "hashrate observation" is inferred from each block time.

    Eg, suppose our hashrate estimate before the last block B was H, and thus our difficulty D was proportional to H, intended to yield (on average) a 10-minute block.  But suppose in fact
    block B was mined after only 2 minutes.  Then we infer that during those 2 minutes, hashrate was ~5H, and update our next block's hashrate estimate (and thus difficulty) upwards accordingly.

    In particular, blocks twice as long get twice the weight: a 1-second block tells us hashrate was (probably) high for only 1 second, but a 24-hour block tells us hashrate was (probably) low
    for a full day - the latter *should* get much more weight in our "recent hashrate" estimate."""

    block_time          = states[-1].timestamp - states[-2].timestamp
    block_time          = max(IDEAL_BLOCK_TIME / 100, min(100 * IDEAL_BLOCK_TIME, block_time))          # Crudely dodge problems from ~0/negative/huge block times
    old_hashrate_est    = TARGET_1 / bits_to_target(states[-1].bits)                                    # "Hashrate estimate" - aka difficulty!
    block_weight        = 1 - math.exp(-block_time / window)                                            # Weight of last block_time seconds, according to exp moving avg
    block_hashrate_est  = (IDEAL_BLOCK_TIME / block_time) * old_hashrate_est                            # Eg, if a block takes 2 min instead of 10, we est hashrate was ~5x higher than predicted
    new_hashrate_est    = (1 - block_weight) * old_hashrate_est + block_weight * block_hashrate_est     # Simple weighted avg of old hashrate est, + block's adjusted hashrate est
    new_target          = round(TARGET_1 / new_hashrate_est)
    return target_to_bits(new_target)

def next_bits_ema2(msg, window):
    # A minor reworking of next_bits_ema() above, meant to produce almost exactly the same numbers in typical cases, but be more resilient to huge/0/negative block times.
    max_prev_timestamp = max(state.timestamp for state in states[-100:-1])
    block_time = max(min(IDEAL_BLOCK_TIME, window) / 100, states[-1].timestamp - max_prev_timestamp)    # Luckily our target formula is ~flat near 0, so can floor block_time at some small val
    old_target = bits_to_target(states[-1].bits)
    new_target = round(old_target / (1 - math.expm1(-block_time / window) * (IDEAL_BLOCK_TIME / block_time - 1)))
    return target_to_bits(new_target)

def next_bits_ema_int_approx(msg, window):
    # An integer-math simplified approximation of next_bits_ema2() above.
    max_prev_timestamp = max(state.timestamp for state in states[-100:-1])
    block_time = max(0, min(window, states[-1].timestamp - max_prev_timestamp))                         # Need block_time <= window for the linear approx below to work (approximate the above)
    old_target = bits_to_target(states[-1].bits)
    new_target = old_target * window // (window + IDEAL_BLOCK_TIME - block_time)                        # Simplifies the corresponding line above via this approx: for 0 <= x << 1, 1-e**(-x) =~ x
    return target_to_bits(new_target)

def exp_int_approx(x, bits_precision=20):
    """Approximates e**(x / 2**bits_precision) using integer math, returning the answer scaled by the same number of bits as the input.  Eg:
    exp_int_approx(1024,   10) ->    2783       (1024/2**10 = 1; e**1 = 2.718281 =~ 2783/2**10)
    exp_int_approx(3072,   10) ->   20567       (3072/2**10 = 3; e**3 = 20.0855 =~ 20567/2**10)
    exp_int_approx(524288, 20) -> 1728809       (524288/2**20 = 0.5; e**0.5 = 1.6487 =~ 1728809/2**20)"""
    assert type(x) is int, str(type(x))                                             # If we pass in a non-int, something has gone wrong

    h = max(0, int.bit_length(x) - bits_precision + 3)                              # h = the number of times we halve x before using our fancy approximation
    term1, term2 = 3 << (bits_precision + h), 3 << (2 * (bits_precision + h))       # Terms from the hairy but accurate approximation we're using - see https://math.stackexchange.com/a/56064
    hth_square_root_of_e_x = (((x + term1)**2 + term2) << (2 * bits_precision)) // ((x - term1)**2 + term2)

    e_x = hth_square_root_of_e_x                                                    # Now just need to square hth_square_root_of_e_x h times, while repeatedly dividing out our scaling factor
    for i in range(h):
        e_x = e_x**2 >> (2 * bits_precision)
    return e_x >> bits_precision                                                    # And finally, we still have one extra scaling factor to divide out.

def next_bits_ema_int_approx2(msg, window):
    # An integer-math version of next_bits_ema2() above, trying to retain the correct exponential behavior for very long block times.
    max_prev_timestamp = max(state.timestamp for state in states[-100:-1])
    block_time = max(min(IDEAL_BLOCK_TIME, window) // 100, states[-1].timestamp - max_prev_timestamp)
    old_target = bits_to_target(states[-1].bits)
    bits_precision = 20
    scaling = 1 << bits_precision
    new_target = scaling**2 * old_target // (scaling**2 - (exp_int_approx(scaling * -block_time // window, bits_precision) - scaling) * (scaling * IDEAL_BLOCK_TIME // block_time - scaling))
    return target_to_bits(new_target)

def next_bits_simple_exponential(msg, window):
    # Dead simple: if the block time is IDEAL_BLOCK_TIME, target is unchanged; if it's more (or less) by n (-n) minutes, scale target by e**(n/window).
    # One nice thing about this is it avoids any need for special handling of huge/0/negative block times.  Eg, successive block times of (-1000000, 1000020) (or vice versa) result in
    # *exactly* the same target as (10, 10).  (This is in fact the only algo with this property!)
    block_time = states[-1].timestamp - states[-2].timestamp
    old_target = bits_to_target(states[-1].bits)
    new_target = round(math.exp((block_time - IDEAL_BLOCK_TIME) / window) * old_target)
    return target_to_bits(new_target)

def next_bits_simple_exponential_int_approx(msg, window):
    # An integer-math version of next_bits_simple_exponential() above.
    block_time = states[-1].timestamp - states[-2].timestamp
    old_target = bits_to_target(states[-1].bits)
    bits_precision = 20
    scaling = 1 << bits_precision
    new_target = exp_int_approx(scaling * (block_time - IDEAL_BLOCK_TIME) // window, bits_precision) * old_target // scaling
    return target_to_bits(new_target)

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
    bits = algo.next_bits(msg, **algo.params)
    target = bits_to_target(bits)
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

    chainwork = states[-1].chainwork + bits_to_work(bits)

    # add a state
    states.append(State(states[-1].height + 1, wall_time, timestamp,
                        bits, chainwork, fx, hashrate, rev_ratio,
                        var_frac, memory_frac, greedy_frac, ' / '.join(msg)))

Algo = namedtuple('Algo', 'next_bits params')

Algos = {
    'k-1' : Algo(next_bits_k, {
        'mtp_window': 6,
        'high_barrier': 60 * 128,
        'target_raise_frac': 64,   # Reduce difficulty ~ 1.6%
        'low_barrier': 60 * 30,
        'target_drop_frac': 256,   # Raise difficulty ~ 0.4%
        'fast_blocks_pct': 95,
    }),
    'k-2' : Algo(next_bits_k, {
        'mtp_window': 4,
        'high_barrier': 60 * 55,
        'target_raise_frac': 100,   # Reduce difficulty ~ 1.0%
        'low_barrier': 60 * 36,
        'target_drop_frac': 256,   # Raise difficulty ~ 0.4%
        'fast_blocks_pct': 95,
    }),
    'd-1' : Algo(next_bits_d, {}),
    'cw-072' : Algo(next_bits_cw, {
        'block_count': 72,
    }),
    'cw-108' : Algo(next_bits_cw, {
        'block_count': 108,
    }),
    'cw-144' : Algo(next_bits_cw, {
        'block_count': 144,
    }),
    'cw-288' : Algo(next_bits_cw, {
        'block_count': 288,
    }),
    'cw-576' : Algo(next_bits_cw, {
        'block_count': 576,
    }),
    'cw-sha-16' : Algo(next_bits_sha, {}),
    'cw-180' : Algo(next_bits_cw, {
        'block_count': 180,
    }),
    'wt-144' : Algo(next_bits_wt, {
        'block_count': 144*2
    }),
    'wt-190' : Algo(next_bits_wt, {
        'block_count': 190*2
    }),
    'wt-288' : Algo(next_bits_wt, {
        'block_count': 288*2
    }),
    'wt-576' : Algo(next_bits_wt, {
        'block_count': 576*2
    }),
    'dgw3-024' : Algo(next_bits_dgw3, { # 24-blocks, like Dash
        'block_count': 24,
    }),
    'dgw3-144' : Algo(next_bits_dgw3, { # 1 full day
        'block_count': 144,
    }),
    'meng-1' : Algo(next_bits_m2, { # mengerian_algo_1
        'window_1': 71,
        'window_2': 137,
    }),
    'meng-2' : Algo(next_bits_m4, { # mengerian_algo_2
        'window_1': 13,
        'window_2': 37,
        'window_3': 71,
        'window_4': 137,
    }),
    # runs wt-144 in external program, compares with python implementation.
    'wt-144-compare' : Algo(next_bits_wt_compare, {
        'block_count': 144
    }),
    'ema-30min' : Algo(next_bits_ema, { # Exponential moving avg
        'window': 30 * 60,
    }),
    'ema-3h' : Algo(next_bits_ema, {
        'window': 3 * 60 * 60,
    }),
    'ema-1d' : Algo(next_bits_ema, {
        'window': 24 * 60 * 60,
    }),
    'ema2-1d' : Algo(next_bits_ema2, {
        'window': 24 * 60 * 60,
    }),
    'emai-1d' : Algo(next_bits_ema_int_approx, {
        'window': 24 * 60 * 60,
    }),
    'emai2-1d' : Algo(next_bits_ema_int_approx2, {
        'window': 24 * 60 * 60,
    }),
    'simpexp-1d' : Algo(next_bits_simple_exponential, {
        'window': 24 * 60 * 60,
    }),
    'simpexpi-1d' : Algo(next_bits_simple_exponential_int_approx, {
        'window': 24 * 60 * 60,
    }),
    'lwma-072' : Algo(next_bits_lwma, {
        'n': 72*2,
    }),
    'lwma-144' : Algo(next_bits_lwma, {
        'n': 144*2,
    }),
    'lwma-190' : Algo(next_bits_lwma, {
        'n': 190*2,
    }),
    'lwma-240' : Algo(next_bits_lwma, {
        'n': 240*2,
    }),
    'lwma-288' : Algo(next_bits_lwma, {
        'n': 288*2,
    }),
    'lwma-576' : Algo(next_bits_lwma, {
        'n': 576*2,
    }),
    'asert-072' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 72),
    }),
    'asert-144' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 144),
    }),
    'asert-208' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 208),
    }),
    'asert-288' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 288),
    }),
    'asert-342' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 342),
    }),
    'asert-407' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 407),
    }),
    'asert-484' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 484),
    }),
    'asert-576' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 576),
    }),
    'asert-685' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 685),
    }),
    'asert-815' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 815),
    }),
    'asert-969' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 969),
    }),
    'asert-1152' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 1152),
    }),
    'asert-2304' : Algo(next_bits_asert, {
        'tau': (IDEAL_BLOCK_TIME * 2304),
    }),
    'asertd-144' : Algo(next_bits_asert_discrete, {
        'window': (IDEAL_BLOCK_TIME * 144),
        'granularity': 100,
    }),
    'aserti1-144' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 144),
        'mode': 1,
    }),
    'aserti1-288' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 288),
        'mode': 1,
    }),
    'aserti1-576' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 576),
        'mode': 1,
    }),
    'aserti2-144' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 144),
        'mode': 2,
    }),
    'aserti2-288' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 288),
        'mode': 2,
    }),
    'aserti2-576' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 576),
        'mode': 2,
    }),
    'aserti3-072' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME *  72),
        'mode': 3,
    }),
    'aserti3-144' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 144),
        'mode': 3,
    }),
    'aserti3-200' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 200),
        'mode': 3,
    }),
    'aserti3-208' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 208),
        'mode': 3,
    }),
    'aserti3-288' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 288),
        'mode': 3,
    }),
    'aserti3-416' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 416),
        'mode': 3,
    }),
    'aserti3-576' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 576),
        'mode': 3,
    }),
    'aserti3-2d' : Algo(next_bits_aserti, {
        'tau': 2*24*3600,
        'mode': 3,
    }),
    'aserti3-mo3-072' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME *  72),
        'mode': 3, 'mo3':True,
    }),
    'aserti3-mo3-144' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 144),
        'mode': 3, 'mo3':True,
    }),
    'aserti3-mo3-200' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 200),
        'mode': 3, 'mo3':True,
    }),
    'aserti3-mo3-208' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 208),
        'mode': 3, 'mo3':True,
    }),
    'aserti3-mo3-288' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 288),
        'mode': 3, 'mo3':True,
    }),
    'aserti3-mo3-416' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 416),
        'mode': 3, 'mo3':True,
    }),
    'aserti3-mo3-576' : Algo(next_bits_aserti, {
        'tau': int(math.log(2) * IDEAL_BLOCK_TIME * 576),
        'mode': 3, 'mo3':True,
    }),
    'wtema-072' : Algo(next_bits_wtema, {
        'alpha_recip': 72, # floor(1/(1 - pow(.5, 1.0/72))), # half-life = 72
    }),
    'wtema-144' : Algo(next_bits_wtema, {
        'alpha_recip': 144, 
    }),
    'wtema-288' : Algo(next_bits_wtema, {
        'alpha_recip': 288,
    }),
    'wtema-576' : Algo(next_bits_wtema, {
        'alpha_recip': 576,
    }),
    'wtema-mo3-072' : Algo(next_bits_wtema, {
        'alpha_recip': 72, 'mo3':1,
    }),
    'wtema-mo3-144' : Algo(next_bits_wtema, {
        'alpha_recip': 144, 'mo3':1,
    }),
    'wtema-mo3-288' : Algo(next_bits_wtema, {
        'alpha_recip': 288, 'mo3':1,
    }),
    'wtema-mo3-576' : Algo(next_bits_wtema, {
        'alpha_recip': 576, 'mo3':1,
    }),
    'wtema-mo3bad-576' : Algo(next_bits_wtema, {
        'alpha_recip': 576, 'mo3':2,
    }),
    'grasberg-144' : Algo(next_bits_grasberg, {
        'nblocks': 144,
    }),
    'grasberg-288' : Algo(next_bits_grasberg, {
        'nblocks': 288,
    }),
    'grasberg-416' : Algo(next_bits_grasberg, {
        'nblocks': 416,
    }),
    'grasberg-576' : Algo(next_bits_grasberg, {
        'nblocks': 576,
    }),
    'grasberg-neutral-144' : Algo(next_bits_grasberg, {
        'nblocks': 144,
        'genesis_time': default_params['INITIAL_TIMESTAMP'] - IDEAL_BLOCK_TIME*default_params['INITIAL_HEIGHT']
    }),
    'grasberg-neutral-288' : Algo(next_bits_grasberg, {
        'nblocks': 288,
        'genesis_time': default_params['INITIAL_TIMESTAMP'] - IDEAL_BLOCK_TIME*default_params['INITIAL_HEIGHT']
    }),
    'grasberg-neutral-416' : Algo(next_bits_grasberg, {
        'nblocks': 416,
        'genesis_time': default_params['INITIAL_TIMESTAMP'] - IDEAL_BLOCK_TIME*default_params['INITIAL_HEIGHT']
    }),
    'grasberg-neutral-576' : Algo(next_bits_grasberg, {
        'nblocks': 576,
        'genesis_time': default_params['INITIAL_TIMESTAMP'] - IDEAL_BLOCK_TIME*default_params['INITIAL_HEIGHT']
    }),
    'grasberg-nodrift-144' : Algo(next_bits_grasberg, {
        'nblocks': 144, 'correct_drift': False,
    }),
    'grasberg-nodrift-288' : Algo(next_bits_grasberg, {
        'nblocks': 288, 'correct_drift': False,
    }),
    'grasberg-nodrift-416' : Algo(next_bits_grasberg, {
        'nblocks': 416, 'correct_drift': False,
    }),
    'grasberg-nodrift-576' : Algo(next_bits_grasberg, {
        'nblocks': 576, 'correct_drift': False,
    })
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
                          params['INITIAL_BCC_BITS'], bits_to_work(params['INITIAL_BCC_BITS']) * (n + N + 1),
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
