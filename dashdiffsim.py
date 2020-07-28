#!/usr/bin/pypy
import random, traceback, platform, sys
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from math import *
import json
import time


import mining

sys.setrecursionlimit(100000)

MAX_BLOCKS = 50000

params = {'algo':['grin-60-3', 'wtema-072'],
          'scenario':'default',
          'num_blocks':4000,

          'INITIAL_BCC_BITS':0x18084bb7,
          'INITIAL_SWC_BITS':0x18013ce9,
          'INITIAL_FX':0.15,
          'INITIAL_TIMESTAMP':1503430225,
          'INITIAL_HASHRATE':1000,    # In PH/s.
          'INITIAL_HEIGHT':481824,
          'BTC_fees':0.02,
          'BCH_fees':0.002,

          # Steady hashrate mines the BCC chain all the time.  In PH/s.
          'STEADY_HASHRATE':300,

          # Variable hash is split across both chains according to relative
          # revenue.  If the revenue ratio for either chain is at least 15%
          # higher, everything switches.  Otherwise the proportion mining the
          # chain is linear between +- 15%.
          'VARIABLE_HASHRATE':10000, # In PH/s.
          'VARIABLE_PCT':15,        # 85% to 115%
          'VARIABLE_WINDOW':2,      # No of blocks averaged to determine revenue ratio
          'VARIABLE_EXPONENT':.5,     # Hashrate = k * f(rev_ratio) ** exponent
          'MEMORY_GAIN':.005,         # if rev_ratio**POWER is 1.01, then the next block's HR will be 0.01*MEMORY_GAIN higher

          # Greedy hashrate switches chain if that chain is more profitable for
          # GREEDY_WINDOW BCC blocks.  It will only bother to switch if it has
          # consistently been GREEDY_PCT more profitable.
          'GREEDY_HASHRATE':0,     # In PH/s.
          'GREEDY_PCT':3,
          'GREEDY_WINDOW':1,
          }

debug=3

seed = 100 #random.randint(0, 2**32-2) # you can also set this to a specific integer for repeatability
print("Seed = %i" % seed)

class Nothing:
    pass

def normalize(data):
  avg = sum(data)/len(data)
  return list(map(lambda x: x/avg, data))

if __name__ == "__main__":

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    def run_round(params, seed):
        runparams = {}
        runparams.update(params)
        runparams['scenario'] = mining.Scenarios[params['scenario']]
        dataframes = []
        params['algo'].sort()
        print(params)
        for i in range(len(params['algo'])):
          random.seed(seed)
          runparams['algo'] = mining.Algos[params['algo'][i]]
          states = mining.run_one_simul(print_it=False, returnstate=True, params=runparams)
          df = {}
          df['name'] = params['algo'][i]
          df['heights']      = [state.height for state in states]
          df['timestamps']   = [state.timestamp for state in states]
          df['wall_times']   = [state.wall_time for state in states]
          df['fxs']          = [state.fx for state in states]
          df['chainworks']   = [state.chainwork for state in states]
          df['hashrates']    = [state.hashrate for state in states]
          df['rev_ratios']   = [1/state.rev_ratio for state in states]
          df['bits']         = [state.bits for state in states]
          df['difficulties'] = [mining.TARGET_1 / mining.bits_to_target(state.bits) for state in states]
          df['greedy_fracs'] = [state.greedy_frac for state in states]
          df['var_fracs']    = [state.var_frac for state in states]
          #df['states'] = states
          dataframes.append(df)
        return dataframes

    print(params)
    app.layout = html.Div(children=[
        html.H1(children="jtoomim's difficulty simulation (alpha version)"),

        html.Div([
          html.H6("Difficulty adjustment algorithm", style={'marginRight': '1em'}),
          dcc.Checklist(
            id='algo-dropdown',
            options=[{'label': algo, 'value':algo} for algo in mining.Algos.keys() if not "compare" in algo],
            value=[algo for algo in params['algo'] ],
            labelStyle={'display': 'inline-block'},
            persistence=True,
            style={'width':'40%', 'verticalAlign':'middle'},),
          ],
          style={'display':'flex'}),
        html.Div([
          html.H6("Scenario", style={'marginRight': '1em'}),
          dcc.RadioItems(
            id='scenario-dropdown',
            options=[{'label': scene, 'value':scene} for scene in mining.Scenarios.keys()],
            value=params['scenario'],
            labelStyle={'display': 'inline-block'},
            persistence=True,
            style={'width':'40%', 'verticalAlign':'middle'},),
          ],
          style={'display':'flex'}),


         #html.Div(dcc.Input(type="checkbox", id="use_lines"), "Use lines"),
        html.Div(id='results_of_run', style={'display':'none'}),
        dcc.Graph(
            id='diff-graph',
            figure={}
        ),
        dcc.Graph(
            id='hashrate-graph',
            figure={}
        ),
        dcc.Graph(
            id='revratio-graph',
            figure={},
        ),        
        # dcc.Graph(
        #     id='intervals-graph',
        #     figure={},
        # ),
        dcc.Graph(
            id='conftimes-graph',
            figure={},
        ),

        html.Div(id='profits', children=[]),

        html.Div(children=(html.H4("Advanced configuration"))),
        html.Div(children=(dcc.Input(id='Seed', value=seed, type="number", name="Randomness seed"),
                           html.H6("Randomness seed")),
                 style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='blocks', value=params['num_blocks'], type="number", 
                               step=100, min=100, max=50000, required=True, persistence=True),
                           html.H6("Number of blocks to simulate")), style={'display':'flex'}),

        # html.Div(children=(dcc.Input(id='', value=params[''], type="number", 
        #                        step=100, min=100, max=50000, required=True, persistence=True),
        #                    html.H6("")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='initial_fx', value=params['INITIAL_FX'], type="number",
                               step=0.01, min=0.0001, max=10, required=True, persistence=True),
                           html.H6("Initial BCH/BTC exchange rate")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='btc_fees', value=params['BTC_fees'], type="number",
                               step=0.01, min=0, max=12.5, required=True, persistence=True),
                           html.H6("Maximum BTC fees (randomized per block)")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='bch_fees', value=params['BCH_fees'], type="number",
                               step=0.01, min=0, max=12.5, required=True, persistence=True),
                           html.H6("Maximum BCH fees (randomized per block)")), style={'display':'flex'}),

        html.Div(children=(html.H6("Consistent miners"), "Consistent miners never leave BCH, no matter how much money they make or lose.")),
        html.Div(children=(dcc.Input(id='steady_hashrate', value=params['STEADY_HASHRATE'], type="number",
                               step=100, min=0, max=500000, required=True, persistence=True),
                           html.H6("Hashrate of consistent miners (PH/s)")), style={'display':'flex'}),
        html.Div(children=(html.H6("Variable miners"),
                           "Variable miners split their hashrate between BCH and BTC depending on the ratio of "
                           "profitability over the last N blocks. Formula:\n",
                           "var_fraction = ((1+var_pct/100) - mean_rev_ratio^var_exponent) * scale_factor\n")),
        html.Div(children=(dcc.Input(id='variable_hashrate', value=params['VARIABLE_HASHRATE'], type="number",
                               step=500, min=0, max=500000, required=True, persistence=True),
                           html.H6("Hashrate of variable miners")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='variable_pct', value=params['VARIABLE_PCT'], type="number",
                               step=1, min=0, max=100, required=True, persistence=True),
                           html.H6("var_pct: The profitability percent at which to allocate 100% of variable hashrate")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='variable_exponent', value=params['VARIABLE_EXPONENT'], type="number",
                               step=0.1, min=0, max=100, required=True, persistence=True),
                           html.H6("var_exponent: higher values create more aggressive switching and more oscillations")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='variable_window', value=params['VARIABLE_WINDOW'], type="number",
                               step=1, min=0, max=100, required=True, persistence=True),
                           html.H6("var_window: Number of blocks averaged to determine revenue ratio")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='memory_gain', value=params['MEMORY_GAIN'], type="number",
                               step=0.001, min=0, max=1, required=True, persistence=True),
                           html.H6("memory_gain: What proportion of variable hashrate will stick around even if profitability falls next block")), style={'display':'flex'}),


        html.Div(children=(html.H6("Greedy miners"),
                           "Greedy miners switch all of their hashrate onto BCH if profitability over the last N blocks exceeds their threshold.")),
        html.Div(children=(dcc.Input(id='greedy_hashrate', value=params['GREEDY_HASHRATE'], type="number",
                               step=500, min=0, max=500000, required=True, persistence=True),
                           html.H6("Hashrate of greedy miners")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='greedy_pct', value=params['GREEDY_PCT'], type="number",
                               step=1, min=0, max=100, required=True, persistence=True),
                           html.H6("greedy_pct: The greedy miners' threshold for hashing")), style={'display':'flex'}),
        html.Div(children=(dcc.Input(id='greedy_window', value=params['GREEDY_WINDOW'], type="number",
                               step=1, min=0, max=100, required=True, persistence=True),
                           html.H6("greedy_window: Number of blocks averaged to determine revenue ratio")), style={'display':'flex'}),



    ])

    @app.callback(
        Output(component_id='results_of_run',     component_property='children'),
        [Input(component_id='algo-dropdown',      component_property='value'),
         Input(component_id='scenario-dropdown',  component_property='value'),
         Input(component_id='blocks',             component_property='value'),
         Input(component_id='initial_fx',         component_property='value'),
         Input(component_id='btc_fees',           component_property='value'),
         Input(component_id='bch_fees',           component_property='value'),
         Input(component_id='steady_hashrate',    component_property='value'),
         Input(component_id='variable_hashrate',  component_property='value'),
         Input(component_id='variable_pct',       component_property='value'),
         Input(component_id='variable_window',    component_property='value'),
         Input(component_id='variable_exponent',  component_property='value'),
         Input(component_id='memory_gain',        component_property='value'),
         Input(component_id='greedy_hashrate',    component_property='value'),
         Input(component_id='greedy_pct',         component_property='value'),
         Input(component_id='greedy_window',      component_property='value'),
         Input(component_id='Seed',               component_property='value'),
         ])
    def update_results_of_run(algo, scenario, num_blocks, initial_fx, btc_fees,
                              bch_fees, steady_hashrate, variable_hashrate,
                              variable_pct, variable_window, variable_exponent,
                              memory_gain, greedy_hashrate, greedy_pct,
                              greedy_window, seed):
        if not type(num_blocks) == int:
          num_blocks = 1000
        elif num_blocks < 100:
          num_blocks = 100
        elif num_blocks > MAX_BLOCKS:
          num_blocks = MAX_BLOCKS

        if not type(initial_fx) in (int, float):
          initial_fx = 0.18
        if not type(btc_fees) in (int, float):
          btc_fees = 0.02
        if not type(bch_fees) in (int, float):
          bch_fees = 0.002
        if not type(steady_hashrate) in (int, float):
          steady_hashrate = 300
        if not type(variable_hashrate) in (int, float):
          variable_hashrate = 0
        if not type(variable_pct) in (int, float):
          variable_pct = 15
        if not type(variable_window) == int:
          variable_window = 6
        if not type(memory_gain) in (int, float):
          memory_gain = 0.005
        if not type(greedy_hashrate) in (int, float):
          greedy_hashrate = 0
        if not type(greedy_pct) in (int, float):
          greedy_pct = 8
        if not type(greedy_window) == int:
          greedy_window = 6
        runparams = {}
        runparams.update(params)
        runparams['algo'] = algo
        runparams['scenario'] = scenario
        runparams['num_blocks'] = num_blocks
        runparams['INITIAL_FX'] = initial_fx
        runparams['STEADY_HASHRATE'] = steady_hashrate
        runparams['VARIABLE_HASHRATE'] = variable_hashrate
        runparams['VARIABLE_PCT'] = variable_pct
        runparams['VARIABLE_WINDOW'] = variable_window
        runparams['VARIABLE_EXPONENT'] = variable_exponent
        runparams['MEMORY_GAIN'] = memory_gain
        runparams['GREEDY_HASHRATE'] = greedy_hashrate
        runparams['GREEDY_PCT'] = greedy_pct
        runparams['GREEDY_WINDOW'] = greedy_window

        t0 = time.time()
        df = run_round(runparams, seed)
        t1 = time.time()
        print("sim_time: %5.3f sec" % (t1-t0))
        dump = json.dumps(df)
        t2 = time.time()
        print("dump_time: %5.3f sec" % (t2-t1))
        return dump

    @app.callback(Output(component_id='diff-graph', component_property='figure'),
                  [Input(component_id='results_of_run', component_property='children')])
    def update_diff_graph(pickled_results):
        dfs = json.loads(pickled_results)
        datalist = []
        for i in range(len(dfs)):
          df = dfs[i]
          name = df['name']
          datalist.append({'x': list(map(lambda x: (x-df['wall_times'][0])/3600/24, df['wall_times'])), 
                          'y': df['difficulties'], 
                          'mode':'lines', 'name':name})
        return {'data': datalist,
               'layout': {'title': "Difficulty", 'xaxis': {'title':'Days since start'}, 'yaxis': {'title':"Difficulty"}}}
    @app.callback(Output(component_id='hashrate-graph', component_property='figure'),
                  [Input(component_id='results_of_run', component_property='children')])
    def update_hashrate_graph(pickled_results):
        t0 = time.time()
        dfs = json.loads(pickled_results)
        t1 = time.time()
        print("decode_time: %5.3f sec" % (t1-t0))
        datalist = []
        for i in range(len(dfs)):
          df = dfs[i]
          name = df['name']
          datalist.append({'x': list(map(lambda x: (x-df['wall_times'][0])/3600/24, df['wall_times'])), 
                          'y': df['hashrates'], 
                          'mode':'lines', 'name':name,})# 'line':{'color':'blue'}}
        return {'data': datalist,
               'layout': {'title': "BCH Hashrate", 'xaxis': {'title':'Days since start'}, 'yaxis': {'title':"Hashrate", "type":"log"}}}


    @app.callback(Output(component_id='revratio-graph', component_property='figure'),
                  [Input(component_id='results_of_run', component_property='children')])
    def update_rev_ratio_graph(pickled_results):
        dfs = json.loads(pickled_results)
        datalist = []
        for i in range(len(dfs)):
          df = dfs[i]
          name = df['name']
          datalist.append({'x': list(map(lambda x: (x-df['heights'][0]), df['heights'])), 
                          'y': df['rev_ratios'], 
                          'mode':'lines', 'name':name})
        datalist.append({'x': list(map(lambda x: (x-dfs[0]['heights'][0]), dfs[0]['heights'])), 
                          'y': normalize(dfs[0]['fxs']), 
                          'mode':'lines', 'name':'Exchange rate', 'line':{'color':'black'}})
        return {'data': datalist,
               'layout': {'title': "Revenue ratio", 'xaxis': {'title':'Block height'}, 'yaxis': {'title':"BCH/BTC revenue ratio"}}}

    # @app.callback(Output(component_id='intervals-graph', component_property='figure'),
    #               [Input(component_id='results_of_run', component_property='children')])
    def update_interval_histogram(pickled_results):
        dfs = json.loads(pickled_results)
        datalist = []
        steps = 20 #int(len(df['wall_times'])**.5 + .5)
        #intervals = [i*3600/steps for i in range(1, steps)]
        start = 6
        stop = 10800
        exp = (stop/start)**(1/steps)
        intervals = [0] + [(stop/exp**steps) * exp**i for i in range(0, steps)]
        for df in dfs:
          idealbins = [len(df['wall_times']) * ((1-e**-(intervals[i]/600)) - (1-e**-(intervals[i-1]/600))) for i in range(1, steps+1)]
          idealbins[-1] = len(df['wall_times']) * (e**-(intervals[-1]/600))
          unitybins = [1. for i in range(1, steps+1)]
          blocktimes = [df['wall_times'][i] - df['wall_times'][i-1] for i in range(1, len(df['wall_times']))]
          blocktimes.sort()
          c = 0
          i = 0
          below_60_sec = 0
          above_1800_sec = 0
          bins = [0]*steps
          for bt in blocktimes:
            if bt < 60:
              below_60_sec += 1
            if bt > 1800:
              above_1800_sec += 1
            while i < (steps-1) and bt > intervals[i]:
              bins[i] = c / idealbins[i]
              i += 1
              c = 0
            c += 1
          bins[i] = c
          intervalcenters = [(interval*(1/exp)) for interval in intervals]
          print(df['name'], below_60_sec, above_1800_sec)
          datalist.append({'x': intervalcenters, 
                          'y': bins, 
                          'name':df['name']})
        datalist.append({'x': intervalcenters, 
                          'y': unitybins, 
                          'name':'Ideal', 'line':{'color':'black'}})

        return {'data': datalist,
               'layout': {'title': "Relative frequency of block intervals", 
                          'xaxis': {'title':'Block interval (sec)', 'type':'log'}, 
                          'yaxis': {'title':"Bias in number of observations", 'type':'log', 'range':[-.5, 2.0]}}}

    @app.callback(Output(component_id='conftimes-graph', component_property='figure'),
                  [Input(component_id='results_of_run', component_property='children')])
    def update_conftimes_graph(pickled_results):
        dfs = json.loads(pickled_results)
        datalist = []
        def conftimesSMA(wnd, df):
          tsdiffs = [df['wall_times'][i+1] - df['wall_times'][i] for i in range(len(df['wall_times'])-1)]
          return [ sum([d*d for d in tsdiffs[i:i+wnd]]) / (df['wall_times'][i+wnd] - df['wall_times'][i]) / 60 / 2 for i in range(len(tsdiffs)-wnd)]


        for i in range(len(dfs)):
          df = dfs[i]
          name = df['name']
          smasize = int(max(24, min(2016, len(df['wall_times'])/50)))
          datalist.append({'x': list(map(lambda x: (x-df['wall_times'][0])/3600/24, df['wall_times'])), 
                          'y': conftimesSMA(smasize, df), 
                          'mode':'lines', 'name':name})
        return {'data': datalist,
               'layout': {'title': "Avg confirmation time (%s block moving average)" % smasize, 
                          'xaxis': {'title':'Days since start'}, 
                          'yaxis': {'title':"Confirmation time (minutes)", 'range':[0, 60.0]}}}

    @app.callback(Output(component_id='profits', component_property='children'),
                  [Input(component_id='results_of_run', component_property='children')])
    def update_profits(pickled_results):
      dfs = json.loads(pickled_results)
      elements = [html.H4("Block and confirmation times, and profitability of different mining strategies")]

      def blocktimes(df):
        runlength = (df['wall_times'][-1] - df['wall_times'][0])
        return runlength / len(df['wall_times'])
      def conftimes(df):
        tsdiffs = [df['wall_times'][i+1] - df['wall_times'][i] for i in range(len(df['wall_times'])-1)]
        runlength = (df['wall_times'][-1] - df['wall_times'][0])
        return sum([d*d for d in tsdiffs]) / (df['wall_times'][-1] - df['wall_times'][0]) / 2

      if len(dfs[0]['rev_ratios']) < 20000:
        elements.append(html.H6("Warning: These numbers are inaccurate for shorter simulations. It is recommended to simulate at least 20,000 blocks if you are interested in these statistics."))
     
      rows = [html.Tr([html.Th("Algorithm"), html.Th("Avg block interval (sec)"), html.Th("Avg conf time (sec)"), html.Th("Greedy"), html.Th("Variable"), html.Th("Steady"), html.Th("Advantage")])]
      for df in dfs:
        bt = blocktimes(df)
        ct = conftimes(df)

        IBT = mining.IDEAL_BLOCK_TIME
        tsdiffs = [df['wall_times'][i+1] - df['wall_times'][i] for i in range(len(df['wall_times'])-1)]
        tsdiffs.append(tsdiffs[-1])
        timecorrection = sum(tsdiffs)/(IBT*len(tsdiffs))
        greedy_profits = sum([ts/IBT*((1-greedy_frac) + (greedy_frac*rev_ratio)) for greedy_frac, rev_ratio, ts in zip(df['greedy_fracs'], df['rev_ratios'], tsdiffs)])/len(df['greedy_fracs']) / timecorrection
        var_profits    = sum([ts/IBT*((1-var_frac)    + (var_frac   *rev_ratio)) for var_frac,    rev_ratio, ts in zip(df['var_fracs'],    df['rev_ratios'], tsdiffs)])/len(df['var_fracs']) / timecorrection
        steady_profits = sum([ts/IBT*rev_ratio for rev_ratio, ts in zip(df['rev_ratios'], tsdiffs)])/len(df['rev_ratios']) / timecorrection
        best  = max(steady_profits, var_profits, greedy_profits)
        worst = min(steady_profits, var_profits, greedy_profits)
        rows.append(html.Tr([html.Td(df['name']),
          html.Td("%5.2f"%bt),
          html.Td("%5.2f"%ct),
          html.Td("%5.3f%%"%(100*greedy_profits-100)), 
          html.Td("%5.3f%%"%(100*var_profits-100)), 
          html.Td("%5.3f%%"%(100*steady_profits-100)),
          html.Td("%5.3f%%"%(100*(best-worst)))]))

      elements.append(html.Table(rows))
      return elements


    app.run_server(debug=True, host='0.0.0.0')
