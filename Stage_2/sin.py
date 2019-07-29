import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 4))

### 第一个画板 第一个子图
ax = fig.add_subplot(111)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

theta = np.arange(-np.pi, np.pi, 2*np.pi/100)
ax.plot(theta, np.sin(theta))

# plt.style.use('ggplot')
# ax.set_xticks([-1.2, 1.2])
# ax.set_yticks([-1.2, 1.2])

plt.show()
