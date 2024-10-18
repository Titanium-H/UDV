# Name: Plot figures
# Function: See comments above functions
# 
#===========================================================
# Necessary package
import matplotlib.pyplot as plt
#===========================================================
#==========================START============================

# Plot Train and Validation Loss in one figure, model independent
def plot_TVloss_one(plot_x, plot_xlabel, plot_ylabel,
                    plot_y1, plot_y1note,
                    plot_y2, plot_y2note,
                    save_name,
                    line_style = '-', range_decay = 0, fig_size = 12
                   ):
    print(f"The first {range_decay} has been discarded")
    plt.figure(figsize = (fig_size, fig_size))
    plt.plot(range(plot_x-range_decay), plot_y1[-(plot_x-range_decay):], linestyle = line_style, label = plot_y1note)
    plt.plot(range(plot_x-range_decay), plot_y2[-(plot_x-range_decay):], linestyle = line_style, label = plot_y2note)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close('all')

# Plot Train or Validation in one figure, all models
def plot_TorVloss_all(plot_x, plot_xlabel, plot_ylabel,
                      plot_y1, plot_y1note,
                      plot_y2, plot_y2note,
                      plot_y3, plot_y3note,
                      plot_y4, plot_y4note,
                      plot_y5, plot_y5note,
                      plot_y6, plot_y6note,
                      save_name,
                      range_decay = 0, fig_size = 10
                     ):
    print(f"The first {range_decay} has been discarded")
    plt.figure(figsize = (fig_size, fig_size))
    plt.plot(range(plot_x - range_decay), plot_y1[-(plot_x - range_decay):], linestyle = '-', label = plot_y1note)
    plt.plot(range(plot_x - range_decay), plot_y2[-(plot_x - range_decay):], linestyle = ':', label = plot_y2note)
    plt.plot(range(plot_x - range_decay), plot_y3[-(plot_x - range_decay):], linestyle = '-', label = plot_y3note)
    plt.plot(range(plot_x - range_decay), plot_y4[-(plot_x - range_decay):], linestyle = ':', label = plot_y4note)
    plt.plot(range(plot_x - range_decay), plot_y5[-(plot_x - range_decay):], linestyle = '-.', label = plot_y5note)
    plt.plot(range(plot_x - range_decay), plot_y6[-(plot_x - range_decay):], linestyle = '--', label = plot_y6note)
    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close('all')

#===========================END=============================