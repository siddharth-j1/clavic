lines = open('paper.txt').readlines()
print("Total lines:", len(lines))
for i, l in enumerate(lines[718:730]):
    print(718+i, repr(l[:100]))
print("...")
for i, l in enumerate(lines[895:902]):
    print(895+i, repr(l[:100]))
