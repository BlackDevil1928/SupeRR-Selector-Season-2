import argparse
from pathlib import Path
import rr_pressure_model as m
import pandas as pd
names='Rashid Khan|Abdul Samad|Haider Ali|Basil Hameed|Muhammad Farooq|Bashir Ahmad|Mahendra Singh Dhoni|Piyush Chawla|Ishant Sharma|Nathan Bracken|Virat Kohli|Faf du Plessis|Wriddhiman Saha|Rohit Sharma|Shikhar Dhawan|Dinesh Karthik|Pradeep Sangwan|Ravindra Jadeja|Amit Mishra|Dwayne Bravo'.split('|')
parser = argparse.ArgumentParser()
parser.add_argument('zip_path', type=Path)
parser.add_argument('--out', type=Path, default=Path('restricted_results'))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
balls,_=m.load_deliveries(args.zip_path)
data,_=m.crossfit(balls,{'main':(False,5),'strict':(True,5)})
for side, frame in data['main'].items():
    d=frame[frame.name.isin(names)].copy()
    d=d[~(d.name.eq('Abdul Samad') & d.player.ne(7446))]
    ranked=m.rank(d,1,1,1)
    strict=m.rank(data['strict'][side],1,1,1)
    ipl=m.rank(frame[frame.circuit.eq('IPL')],1,1,1)
    ranked['strict_score']=strict.score
    ranked['ipl_score']=ipl.score
    ranked['tier']=1
    ranked.loc[ranked.high_n.ge(36)&ranked.base_n.ge(72)&ranked.matches.ge(8),'tier']=0
    ranked=ranked.sort_values(['tier','score'],ascending=[True,False]).drop_duplicates('name')
    if side == 'bowl':
        fallback_events = d[d.player.eq(147)]
        assert len(fallback_events) and not fallback_events.high.any()
        ranked.loc[147, ['name', 'high_n', 'base_n', 'matches', 'positive_matches', 'tier']] = [
            'Pradeep Sangwan', 0, len(fallback_events), 0, 0, 2
        ]
    ranked.insert(0, 'dropdown_rank', range(1, len(ranked) + 1))
    ranked.to_csv(args.out / f'{side}_restricted.csv')
    cells=d.groupby(['player','high','phase','innings'],observed=True).residual.agg(['size','mean'])
    cells.to_csv(args.out / f'{side}_cells.csv')
    print(side,ranked[['name','score','matches','high_n','positive_matches','tier','strict_score','ipl_score']].to_string())
