import pickle
import matplotlib.pyplot as plt

with open('batch1.pkl', 'rb') as fp:
    bat_dict = pickle.load(fp)

print(bat_dict)
plt.figure(1)
plt.plot(bat_dict['b1c43']['summary']['cycle'], bat_dict['b1c43']['summary']['QD'])
plt.figure(2)
plt.plot([i for i in range(len(bat_dict['b1c43']['cycles']['10']['V']))], bat_dict['b1c43']['cycles']['10']['V'])
print()
plt.show()
