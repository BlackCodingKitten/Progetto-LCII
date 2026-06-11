import csv
import argparse
import sys
import pandas as pd 
import random

random.seed(42)
csv.field_size_limit(sys.maxsize)

def subset_selection (filename, row)->list: 
    subset = []
    c0 = 0
    c1 = 0
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for riga in reader:
                # text--label
            if c0+c1 < row:
                if c0==c1: 
                    match riga["label"]: 
                        case"1":
                            c1+=1
                        case "0":
                            c0+=1
                    subset.append(riga)
                    continue
                elif c0>c1 and riga["label"] =="1":
                    c1+=1
                    subset.append(riga)
                    continue
                elif c0<c1 and riga["label"] == "0":
                    c0+=1
                    subset.append(riga)
                    continue
            else:
                break
    print(f"0: {c0}, 1:{c1}, totale estratti{len(subset)}")
    return subset
                
    
def main (filename: str, row: int, output:str):
    df  = pd.DataFrame(subset_selection(filename,row), columns=["text", "label"])
    df = df.sample(frac=1).reset_index(drop=True)
    # print(df)
    df.to_csv(f"{output}", index=False)
    print(f"File {output} salvato")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Immetti il Nome del file e quanti sottotesti vuoi estrarre")
    
    parser.add_argument("filename", type=str, help="il nome del file da cui vuoi estrarre i dati")
    parser.add_argument("-row", type=int,  help="il numero di righe che vuoi estrarre dal file")
    parser.add_argument("-out", type=str, help="il nome del file di output")
    
    args = parser.parse_args()
    
    print(f"Avvio estrazione di {args.row} da {args.filename}")

    main(args.filename, args.row, args.out)