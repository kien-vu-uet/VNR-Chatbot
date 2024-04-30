import sys, json, csv
from tqdm import tqdm

if __name__ == '__main__':
    csv_path = sys.argv[1]
    # json_path = sys.argv[2]
    # fields_names = sys.argv[2].split('|')
    data = []
    with open(csv_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in tqdm(csv_reader):
            data.append(row)
        f.close()
    print(json.dumps(data, ensure_ascii=False, indent=4))