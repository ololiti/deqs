# this is just so i can plot ||z - f(z)|| over time
import matplotlib.pyplot as plt

eq = "z - f(z) = "
f = open("data.txt", "r")
differences = []
for line in f:
    if line.startswith(eq):
        differences.append(float(line[len(eq):]))
f.close()

timesteps = [i for i in range(len(differences))]

COLORS = ['xkcd:blurple', 'xkcd:lavender', 'xkcd:lightblue', 'xkcd:indigo', 'xkcd:babypink']
plt.figure()
plt.scatter(timesteps, differences, c=COLORS[0])

plt.xlabel('timesteps')
plt.ylabel('||z - f(z)||')
plt.xlim(0, len(timesteps))
plt.ylim(0, 8)
plt.grid()

plt.savefig("fixedpoint.png")
