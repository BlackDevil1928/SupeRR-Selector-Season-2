"""External first-innings IPL pressure check using Cricsheet JSON, 2018–2021."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from rr_pressure_model import score

ALIASES = {
    "Abdul Samad": ["Abdul Samad"], "Ravindra Jadeja": ["RA Jadeja"],
    "Rashid Khan": ["Rashid Khan"], "Virat Kohli": ["V Kohli"],
    "Dinesh Karthik": ["KD Karthik", "D Karthik"], "Rohit Sharma": ["RG Sharma"],
    "Ishant Sharma": ["I Sharma", "Ishant Sharma"], "Piyush Chawla": ["PP Chawla"],
    "Mahendra Singh Dhoni": ["MS Dhoni"], "Pradeep Sangwan": ["PJ Sangwan"],
    "Amit Mishra": ["A Mishra"], "Dwayne Bravo": ["DJ Bravo"],
    "Shikhar Dhawan": ["S Dhawan"], "Wriddhiman Saha": ["WP Saha"],
    "Faf du Plessis": ["F du Plessis"],
}
CREDIT = {"caught", "caught and bowled", "bowled", "lbw", "stumped", "hit wicket"}

def read_matches(path):
    rows = []
    with ZipFile(path) as archive:
        for filename in sorted(archive.namelist()):
            if not filename.endswith(".json"):
                continue
            match = json.loads(archive.read(filename))
            info = match["info"]
            year = int(info["dates"][0][:4])
            innings = match["innings"]
            if (not 2018 <= year <= 2021 or info["overs"] != 20 or len(innings) < 2
                    or innings[1].get("target", {}).get("overs") != 20
                    or info["outcome"].get("method")):
                continue
            team_runs = wickets = used = 0
            positions = {}
            registry = info["registry"]["people"]
            for over in innings[0]["overs"]:
                for ball in over["deliveries"]:
                    batter, non_striker, bowler = (ball[key] for key in ("batter", "non_striker", "bowler"))
                    for name in (batter, non_striker):
                        positions.setdefault(name, len(positions) + 1)
                    extras = ball.get("extras", {})
                    dismissals = [w for w in ball.get("wickets", []) if w["kind"] != "retired hurt"]
                    bat_runs = ball["runs"]["batter"]
                    wide = "wides" in extras
                    noball = "noballs" in extras
                    phase = "PP" if used < 36 else "Middle" if used < 96 else "Death"
                    # Do not infer bowling style from outcome fields in the public JSON.
                    rows.append((filename, used, wickets, team_runs, phase, positions[batter],
                                 registry[batter], batter, registry[bowler], bowler, wide,
                                 not (wide or noball),
                                 bat_runs - 5*sum(w["player_out"] == batter for w in dismissals),
                                 5*sum(w["kind"] in CREDIT for w in dismissals)
                                 - bat_runs - extras.get("wides",0) - extras.get("noballs",0)))
                    team_runs += ball["runs"]["total"]
                    wickets += len(dismissals)
                    used += not (wide or noball)
    return pd.DataFrame(rows,columns=["match","used","wickets","runs","phase","position",
                                      "bat_id","bat_name","bowl_id","bowl_name","wide","legal","bat","bowl"])

def label(frame, par):
    frame = frame.copy()
    limit = frame.phase.map({"PP":3,"Middle":4,"Death":5})
    frame["high"] = frame.used.ge(12) & frame.wickets.ge(limit) & (frame.used.map(par)-frame.runs).ge(6)
    frame["wicket_band"] = (frame.wickets//2).clip(upper=3)
    frame["role"] = pd.cut(frame.position,[0,3,5,7,20],labels=["1-3","4-5","6-7","8+"])
    return frame

def aggregate(balls, side):
    results=[]
    for train_index,test_index in GroupKFold(5).split(balls, groups=balls.match):
        train = balls.iloc[train_index]
        par = train[train.legal].groupby("used").runs.median()
        train,test=label(train,par),label(balls.iloc[test_index],par)
        if side=="bat":
            train,test=train[~train.wide].copy(),test[~test.wide].copy()
        keys=["phase","high","wicket_band"] + (["role"] if side=="bat" else [])
        prediction=np.full(len(test),train[side].mean())
        for size in range(1,len(keys)+1):
            group=keys[:size]
            baseline=train.groupby(group,observed=True)[side].agg(["sum","size"])
            cells=test[group].merge(baseline.reset_index(),on=group,how="left").fillna({"sum":0,"size":0})
            prediction=(cells["sum"].to_numpy()+40*prediction)/(cells["size"].to_numpy()+40)
        test["residual"]=test[side]-prediction
        results.append(test)
    return pd.concat(results)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cricsheet_zip",type=Path)
    parser.add_argument("--out",type=Path,default=Path("external_ipl_results.csv"))
    args=parser.parse_args()
    balls=read_matches(args.cricsheet_zip)
    records=[]
    for side in ("bat","bowl"):
        rated=aggregate(balls,side)
        for name,aliases in ALIASES.items():
            ids=rated.loc[rated[f"{side}_name"].isin(aliases),f"{side}_id"].unique()
            if len(ids)!=1:
                continue
            p=rated[rated[f"{side}_id"].eq(ids[0])]
            high=p[p.high]
            base=p[~p.high]
            matches=high.groupby("match").residual.sum()
            row=dict(player=name,side=side,pressure=len(high),ordinary=len(base),
                     matches=len(matches),positive=int(matches.gt(0).sum()))
            row["rating"]=100*score(high.residual.sum(),len(high),base.residual.sum(),len(base)) if len(high) and len(base) else np.nan
            records.append(row)
    pd.DataFrame(records).to_csv(args.out,index=False,float_format="%.4f")
    print(f"{balls.match.nunique()} matches | {len(balls)} first-innings deliveries")

if __name__=="__main__":
    main()
