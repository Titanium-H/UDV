# Name: Plot figures (Classification)
# Function: See comments above functions
# 
#===========================================================
# Necessary package
import matplotlib.pyplot as plt
#===========================================================
#==========================START============================

# Plot Train and Validation Loss in one figure, model independent
def plot_TVlossacc_one(plot_x1, plot_x1label, plot_y1label,
                       plot_y1, plot_y1note,
                       plot_y2, plot_y2note,
                       plot_x2, plot_x2label, plot_y3label,
                       plot_y3, plot_y3note,
                       plot_y4, plot_y4note,
                       save_name,
                       line_style = '-', range_decay = 0, fig_size = 18
                      ):
    print(f"The first {range_decay} has been discarded")
    fig, axs = plt.subplots(1, 2, figsize=(fig_size, fig_size//2))
    axs[0].plot(range(plot_x1-range_decay), plot_y1[-(plot_x1-range_decay):], linestyle = line_style, label = plot_y1note)
    axs[0].plot(range(plot_x1-range_decay), plot_y2[-(plot_x1-range_decay):], linestyle = line_style, label = plot_y2note)
    axs[0].set_xlabel(plot_x1label)
    axs[0].set_ylabel(plot_y1label)
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(range(plot_x2-range_decay), plot_y3[-(plot_x2-range_decay):], linestyle = line_style, label = plot_y3note)
    axs[1].plot(range(plot_x2-range_decay), plot_y4[-(plot_x2-range_decay):], linestyle = line_style, label = plot_y4note)
    axs[1].set_xlabel(plot_x2label)
    axs[1].set_ylabel(plot_y3label)
    axs[1].legend()
    axs[1].grid(True)  
    
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close('all')

    
# Plot Train or Validation in one figure, all models
def plot_TorVlossacc_all(plot_x1, plot_x1label, plot_y1label,
                         plot_y1, plot_y1note,
                         plot_y2, plot_y2note,
                         plot_y3, plot_y3note,
                         plot_y4, plot_y4note,
                         plot_y5, plot_y5note,
                         plot_y6, plot_y6note,
                         plot_y7, plot_y7note,
                         save_name,
                         range_decay = 0, fig_size = 10
                        ):
    print(f"The first {range_decay} has been discarded")   
    
    plt.figure(figsize = (fig_size, fig_size))
    plt.plot(range(plot_x1 - range_decay), plot_y1[-(plot_x1 - range_decay):], linestyle = '-', label = plot_y1note)
    plt.plot(range(plot_x1 - range_decay), plot_y2[-(plot_x1 - range_decay):], linestyle = ':', label = plot_y2note)
    plt.plot(range(plot_x1 - range_decay), plot_y3[-(plot_x1 - range_decay):], linestyle = '-', label = plot_y3note)
    plt.plot(range(plot_x1 - range_decay), plot_y4[-(plot_x1 - range_decay):], linestyle = ':', label = plot_y4note)
    plt.plot(range(plot_x1 - range_decay), plot_y5[-(plot_x1 - range_decay):], linestyle = '-.', label = plot_y5note)
    plt.plot(range(plot_x1 - range_decay), plot_y6[-(plot_x1 - range_decay):], linestyle = '--', label = plot_y6note)
    plt.plot(range(plot_x1 - range_decay), plot_y7[-(plot_x1 - range_decay):], linestyle = '--', label = plot_y7note)
    plt.xlabel(plot_x1label)
    plt.ylabel(plot_y1label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close('all')

#===========================END=============================