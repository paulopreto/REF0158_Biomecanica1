import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./results/medidas_retofemoral.csv', skiprows=1, delimiter=',', dtype='float')
print(data)

antes = data[0]
depois = data[1]

evolucaomuscle = depois - antes
print(evolucaomuscle)

plt.bar(['antes','depois'],[antes,depois])
plt.show()

print('ACABOU VAMOS EMBORA. BOM FINAL DE SEMANA! TECHAU!')
