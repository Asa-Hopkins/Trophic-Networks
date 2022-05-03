import csv
with open("newtests2.csv", newline='') as f:
    reader = list(csv.reader(f,delimiter=';',quotechar='"'))
    for row in reader:
        if row[0] == 'ID':
            continue
        for a in range(len(row)):
            try:
                row[a] = float(row[a])
            except:
                pass
        layout = row[4]
        connectivity = row[5]
        print(f'ID {row[0]}, acc {row[16]*100:.2f}±{row[17]*100/(10**0.5):.2f}%, inc {row[28]:.4f}±{row[29]:.4f}, loss {row[40]:.0f}±{row[41]:.0f}')
