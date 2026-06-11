import csv
import sys
import argparse
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def keep_first_column() -> None:
    with open("data/subset/test/labeled_test_subset.csv", "r", encoding="utf-8", newline="") as infile, open("data/subset/test_subset.csv", "w", encoding="utf-8", newline="") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            if row:
                writer.writerow([row[0]])


def main():
   

    keep_first_column()


if __name__ == "__main__":
    main()