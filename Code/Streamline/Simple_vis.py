
import wntr
import matplotlib.pyplot as plt

anytown = wntr.network.WaterNetworkModel("Networks/anytown-3.inp")
hanoi = wntr.network.WaterNetworkModel("Networks/hanoi-3.inp")
anytown_end = wntr.network.WaterNetworkModel("Networks/Step_50_Anytown.inp")
hanoi_end = wntr.network.WaterNetworkModel("Networks/Step_50_Hanoi.inp")

# Plot the two networks in a single plot
wntr.graphics.plot_network(hanoi, node_size=10, link_width=0.5)
plt.show()

wntr.graphics.plot_network(anytown, node_size=10, link_width=0.5)
plt.show()

wntr.graphics.plot_network(anytown_end, node_size=10, link_width=0.5)
plt.show()

wntr.graphics.plot_network(hanoi_end, node_size=10, link_width=0.5)
plt.show()

