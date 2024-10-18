# Name: Plot figures
# Function: See comments above functions
# Can be shared to classification tasks
#===========================================================
# Necessary package
import matplotlib.pyplot as plt
import torch
#===========================================================
#==========================START============================
# Plot singular values in one figure (All models)
def plot_SingularValue_all(plot_xlabel, plot_ylabel, num_seeds, 
                           w_m0, u_m0, m0note,
                           w_m1, u_m1, m1note,
                           w_m2, u_m2, m2note,
                           w_m3, u_m3, m3note,
                           l_m4, m4note,
                           l_m5, m5note,
                           save_name,
                           fig_size = 12, plot_yscale = 'log'
                          ):
    
    # Save singular values from all seeds
    SVD_uw_m_0 = []
    SVD_uw_m_1 = []
    SVD_uw_m_2 = []
    SVD_uw_m_3 = []
    SVD_l_m_4 = []
    SVD_l_m_5 = []

    # Index seeds
    seed_index_list = list(range(num_seeds))

    for seed_index in seed_index_list:
        u1w1_m_0 = torch.mul(w_m0[seed_index].t(), u_m0[seed_index]) 
        S_u1w1_m_0 = torch.linalg.svdvals(u1w1_m_0).cpu()

        u1w1_m_1 = torch.mul(w_m1[seed_index].t(), u_m1[seed_index]) 
        S_u1w1_m_1 = torch.linalg.svdvals(u1w1_m_1).cpu()

        u1w1_m_2 = torch.mul(w_m2[seed_index].t(), u_m2[seed_index])
        S_u1w1_m_2 = torch.linalg.svdvals(u1w1_m_2).cpu()

        u1w1_m_3 = torch.mul(w_m3[seed_index].t(), u_m3[seed_index])
        S_u1w1_m_3 = torch.linalg.svdvals(u1w1_m_3).cpu()

        S_u1_m_4 = torch.linalg.svdvals(l_m4[seed_index]).cpu()
        S_u1_m_5 = torch.linalg.svdvals(l_m5[seed_index]).cpu()

        SVD_uw_m_0.append(S_u1w1_m_0)
        SVD_uw_m_1.append(S_u1w1_m_1)
        SVD_uw_m_2.append(S_u1w1_m_2)
        SVD_uw_m_3.append(S_u1w1_m_3)
        SVD_l_m_4.append(S_u1_m_4)
        SVD_l_m_5.append(S_u1_m_5)

    Avg_SVD_uw_m_0 = torch.mean(torch.stack(SVD_uw_m_0), dim = 0)
    Avg_SVD_uw_m_1 = torch.mean(torch.stack(SVD_uw_m_1), dim = 0)
    Avg_SVD_uw_m_2 = torch.mean(torch.stack(SVD_uw_m_2), dim = 0)
    Avg_SVD_uw_m_3 = torch.mean(torch.stack(SVD_uw_m_3), dim = 0)
    Avg_SVD_l_m_4 = torch.mean(torch.stack(SVD_l_m_4), dim = 0)
    Avg_SVD_l_m_5 = torch.mean(torch.stack(SVD_l_m_5), dim = 0)


    plt.figure(figsize=(fig_size, fig_size))
    plt.plot(Avg_SVD_uw_m_0.numpy(), marker = 'o', label = m0note, linestyle='--')
    plt.plot(Avg_SVD_uw_m_1.numpy(), marker = '*', label = m1note, linestyle=':')
    plt.plot(Avg_SVD_uw_m_2.numpy(), marker = 'o', label = m2note, linestyle='--')
    plt.plot(Avg_SVD_uw_m_3.numpy(), marker = '*', label = m3note, linestyle=':')
    plt.plot(Avg_SVD_l_m_4.numpy(), marker = 's', label = m4note, linestyle='-.')
    plt.plot(Avg_SVD_l_m_5.numpy(), marker = '.', label = m5note, linestyle='-')

    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)
    plt.yscale(plot_yscale)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_name)
    plt.close('all')    
    print("Singular value has been plot")
    
#===========================END=============================