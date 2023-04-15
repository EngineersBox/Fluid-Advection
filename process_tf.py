import re

category = "mpirun -np (\d+)"
pattern = "COMP: (\d+\.\d+) COUNT: (\d+)"

def main():
    values = {}
    current = None
    with open('batchAdv4Node.sh.o79495696') as f:
        for line in f:
            result = re.search(category, line)
            if (result):
                current = result.group(1)
                values[current] = [0.0, 0]
                continue
            result = re.search(pattern, line)
            if (result):
                values[current][0] += float(result.group(1)) / float(result.group(2))
                values[current][1] += 1
    overall = 0.0
    for k,v in values.items():
        print("Cores:", k, "Avg: ", v[0] / float(v[1]))
        overall += v[0] / float(v[1])
    print("Average:", overall / 3.0)

if __name__ == '__main__':
    main()
