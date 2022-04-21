# +
import pandas as pd

pd.set_option('display.max_rows', df.shape[0]+1)

def totalTargets(group):
    g = group['ppl'].agg('sum')
    group['total_ppl'] = g
    return group


# +
df = pd.read_json('/mmfs1/gscratch/zlab/margsli/gitfiles/mod/results/hp_sweep_results.jsonl', lines=True)

df.groupby(['num_steps', 'lr','dense', 'out_domain', 'domain']).last()
# -




