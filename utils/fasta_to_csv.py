import pandas as pd
import sys

"""
Converts fasta format to tab-separated .csv
Splits each sequence into a sub-sequence of up to 2000 characters each consisting of 200 10-mers
"""

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

assert(len(sys.argv) == 3)
input_file = sys.argv[1]
output_file = sys.argv[2]

names = []
sequences = []
shortest = 100000
longest = 0

with open(input_file, "r") as file:
    seq = []
    for index,line in enumerate(file):
        if ">" in line:
            _, line = line.split(">", 1) # remove leading '>'
            line, _ = line.split(" ", 1) # remove trailing description
            names.append(line.rstrip())
            if (index == 0):
                continue # the first line is a sequence name so do not append a sequence yet
            sequence = "".join(seq).rstrip()
            #subsequences = list(chunks(sequence, 100))
            #for seq in subsequences:
                #kmers = "".join([seq[i:i+10] for i in range(0, len(seq), 10)]) # join with space to get kmers
                #sequences.append(kmers)
                #names.append(line.rstrip())
                #seq = []
            #kmers = " ".join([sequence[i:i+12] for i in range(0, len(sequence), 12)])
            kmers = " ".join([sequence[i:i+10] for i in range(0, len(sequence), 10)])
            sequences.append(kmers)
            if (len(kmers.split()) < shortest):
                shortest = len(kmers.split())
            if (len(kmers.split()) > longest):
                longest = len(kmers.split())
            seq = []
            continue
        seq.append(line.rstrip())

#sequences.pop(0) # removes the first empty element we've added in the sequence list
print(shortest)
print(longest)

# cut of all sequences that are too long
cut = []
for s in sequences:
    s = s[:shortest]
    cut.append(s)

df = pd.DataFrame(list(zip(names, cut)), columns=['Name', 'Sequence'])
print(df)
df.to_csv(output_file, sep="\t", index=False)
