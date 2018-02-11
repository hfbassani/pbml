import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

label_prop_1 = [0.0455, 0.0074, 0.0125, 0.0068, 0.0101, 0.0500, 0.0091]
label_prop_1err = [0.0107, 0.0050, 0.0085, 0.0058, 0.0015, 0.0130, 0.0055]

label_spreading_1 = [0.0286, 0.0113, 0.0203, 0.0135, 0.0112, 0.0417, 0.0088]
label_spreading_1err = [0.0118, 0.0060, 0.0074, 0.0098, 0.0014, 0.0125, 0.0047]

label_prop_5 = [0.0943, 0.0404, 0.0404, 0.0377, 0.0511, 0.0833, 0.0532]
label_prop_5err = [0.0197, 0.0186, 0.0353, 0.0254, 0.0037, 0.0283, 0.0212]

label_spreading_5 = [0.0606, 0.0451, 0.0561, 0.0406, 0.0500, 0.0978, 0.0515]
label_spreading_5err = [0.0227, 0.0135, 0.0185, 0.0097, 0.0043, 0.0306, 0.0116]

label_prop_10 = [0.1481, 0.0885,0.0856, 0.0609, 0.1008, 0.1395, 0.0886]
label_prop_10err = [0.0291, 0.0212, 0.0305, 0.0144, 0.0091, 0.0384, 0.0205]

label_spreading_10 = [0.0842, 0.0851, 0.0717, 0.0725, 0.1049, 0.1563, 0.1138]
label_spreading_10err = [0.0171, 0.0117, 0.0268, 0.0264, 0.0055, 0.0424, 0.0229]

label_prop_25 = [0.2896, 0.1680, 0.1652, 0.1565, 0.2488, 0.3313, 0.2229]
label_prop_25err = [0.0451, 0.0346, 0.0493, 0.0361, 0.0127, 0.0508, 0.0167]

label_spreading_25 = [0.2155, 0.1953, 0.1871, 0.1778, 0.2550, 0.3164, 0.2616]
label_spreading_25err = [0.0237, 0.0243, 0.0429, 0.0369, 0.0070, 0.0551, 0.0329]

label_prop_50 = [0.7626, 0.6510, 0.3957, 0.3681, 0.6763, 0.7291, 0.4744]
label_prop_50err = [0.0312, 0.0266, 0.0510, 0.1176, 0.0436, 0.0737, 0.0297]

label_spreading_50 = [0.7626, 0.6510, 0.4144, 0.3884, 0.6317, 0.6938, 0.5013]
label_spreading_50err = [0.0312, 0.0266, 0.0652, 0.0804, 0.0453, 0.0624, 0.0321]

label_prop_75 = [0.8030, 0.6693, 0.5562, 0.6164, 0.9709, 0.8458, 0.7067]
label_prop_75err = [0.0283, 0.0253, 0.0495, 0.0341, 0.0051, 0.0751, 0.0472]

label_spreading_75 = [0.8098, 0.6836, 0.5980, 0.6618, 0.9766, 0.8291 , 0.7354]
label_spreading_75err = [0.0253, 0.0270, 0.0387, 0.0224, 0.0031, 0.0769 , 0.0274]

label_prop_100 = [0.8148, 0.7339, 0.6465, 0.6386, 0.9941, 0.8833, 0.9451]
label_prop_100err = [0.0271, 0.0326, 0.0613, 0.0450, 0.0013, 0.0585, 0.0183]

label_spreading_100 = [0.8114, 0.7296, 0.6465, 0.6618, 0.9941, 0.8812, 0.9451]
label_spreading_100err = [0.0264, 0.0295, 0.0641, 0.0407, 0.0013, 0.0594, 0.0183]


ss_som_1 = [0.776094276094, 0.692274305556, 0.393752390889, 0.568115942029, 1, 0.199666123146, 0.226]
ss_som_5 = [0.796296296296, 0.700086805556, 0.394495913754, 0.579710144928, 1, 0.268343815514, 0.229]
ss_som_10 = [0.792929292929, 0.717447916667, 0.396695357329, 0.6038647343, 1, 0.364158707974, 0.229]
ss_som_25 = [0.784511784512, 0.714409722222, 0.45461658842, 0.646376811594, 1, 0.201451976085, 0.229]
ss_som_50 = [0.782828282828, 0.714409722222, 0.531277169188, 0.697584541063, 1, 0.195900302819, 0.224]
ss_som_75 = [0.799663299663, 0.718315972222, 0.580898974092, 0.696618357488, 1, 0.214224706887, 0.224]
ss_som_100 = [0.821548821549, 0.724826388889, 0.598113371587, 0.691787439614, 1, 0.912493205994, 0.233]

ss_som_1err = [0.0506940906563, 0.0282209854189, 0.0859517972289, 0.0434782608694, 1, 0.0903130239877, 0.023]
ss_som_5err = [0.0473779369679, 0.023337823345, 0.0733377286382, 0.039371239731, 1, 0.116265735737, 0.031]
ss_som_10err = [0.0330219616935, 0.0276213586401, 0.0645283476834, 0.0715889219434, 1, 0.233674838508, 0.042]
ss_som_25err = [0.048966463206, 0.0277209229896, 0.100586970492, 0.0458071902298, 1, 0.0858330835358, 0.058]
ss_som_50err = [0.0273147823898, 0.0305365609364, 0.0688967889051, 0.0375974544091, 1, 0.0390518859763, 0.061]
ss_som_75err = [0.0495490325007, 0.0321195218321, 0.0840256317293, 0.0549005707969, 1, 0.052078331187, 0.059]
ss_som_100err = [0.0281200220346, 0.0302015976686, 0.129709997086, 0.0647973591481, 1, 0.0545, 0.069]

titles = ['Breast', 'Diabetes', 'Glass', 'Liver', 'Pendigits', 'Shape', 'Vowel']

for i in xrange(len(titles)):

    current_values_ssom = [ss_som_1[i], ss_som_5[i], ss_som_10[i], ss_som_25[i], ss_som_50[i], ss_som_75[i], ss_som_100[i]]
    current_values_prop = [label_prop_1[i], label_prop_5[i], label_prop_10[i], label_prop_25[i], label_prop_50[i], label_prop_75[i], label_prop_100[i]]
    current_values_spre = [label_spreading_1[i], label_spreading_5[i], label_spreading_10[i], label_spreading_25[i], label_spreading_50[i], label_spreading_75[i], label_spreading_100[i]]

    current_values_ssom_err = [ss_som_1err[i], ss_som_5err[i], ss_som_10err[i], ss_som_25err[i], ss_som_50err[i], ss_som_75err[i],ss_som_100err[i]]
    current_values_prop_err = [label_prop_1err[i], label_prop_5err[i], label_prop_10err[i], label_prop_25err[i], label_prop_50err[i],label_prop_75err[i], label_prop_100err[i]]
    current_values_spre_err = [label_spreading_1err[i], label_spreading_5err[i], label_spreading_10err[i], label_spreading_25err[i],label_spreading_50err[i], label_spreading_75err[i], label_spreading_100err[i]]

    percentage_values = np.linspace(1, 100, num=7)#[1, 5, 10, 25, 50, 75, 100]
    percentage_labels = ['1%', '5%', '10%', '25%', '50%', '75%' , '100%']

    fig, ax = plt.subplots()
    ax.yaxis.grid()
    ax.set_ylim([0, 1])
    ax.set_xticklabels(percentage_labels)

    title = titles[i]
    plt.title(title, fontsize=14)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.xticks(percentage_values)

    plt.errorbar(percentage_values, current_values_ssom, current_values_ssom_err, label='SS-SOM', linestyle='-', marker='o', clip_on=False, markeredgewidth=2, capsize=5)
    plt.errorbar(percentage_values, current_values_spre, current_values_spre_err, label='Label Spreading', linestyle='-', marker='D',clip_on=False, markeredgewidth=2, capsize=5)
    plt.errorbar(percentage_values, current_values_prop, current_values_prop_err, label='Label Propagation', linestyle='-', marker='x', clip_on=False, markeredgewidth=2, capsize=5)
    plt.legend()

    plt.savefig("{0}-wcci.pdf".format(titles[i]), format="pdf")

    # plt.show()

# SSH-SOM @1\% & 0.7828 (0.0330) & 0.7305 (0.0252) & 0.5435 (0.0706) & 0.6550 (0.0253) & 0.8312 (0.0578)
# SSH-SOM @5\% & 0.7811 (0.0331) & 0.7144 (0.0296) & 0.5442 (0.0897) & 0.6560 (0.0199) & 0.8540 (0.0649)
# SSH-SOM @10\% & 0.7862 (0.0267) & 0.7010 (0.0370) & 0.5514 (0.1105) & 0.6647 (0.0239) & 0.8646 (0.0746)
# SSH-SOM @25\% & 0.7946 (0.0313) & 0.6944 (0.0498) & 0.5500 (0.1051) & 0.6734 (0.0424) & 0.8748 (0.0754)
# SSH-SOM @50\% & 0.8081 (0.0321) & 0.6992 (0.0477) & 0.5731 (0.1279) & 0.6812 (0.0597) & 0.8916 (0.0586)
# SSH-SOM @75\% & 0.8115 (0.0253) & 0.7079 (0.0507) & 0.5950 (0.1176) & 0.6956 (0.0498) & 0.9062 (0.0616)
# SSH-SOM @100\% & 0.8283 (0.0239) & 0.7101 (0.0622) & 0.5965 (0.0968) & 0.6985 (0.0710) & 0.9208 (0.0545)


