import h5py  # 3.11.0
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle
'''
.mat --> dict; batch是一批电池的意思
bat_dict{
         'b3ci':{
                'cycle_life': cl, 'charge_policy': policy, 
                'summary': 
                    {
                      'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':summary_TA,
                      'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,'cycle': summary_CY
                    },
                'cycles': # 每个cycle具体的数据
                    {
                     'j in range(cell_i total cycles)' : 
                        {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                    }
                }  
        }
'''
matFilename = './data/2018-04-12_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(matFilename)
# print(list(f.keys()))

batch = f['batch']
# print(list(batch.keys()))

num_cells = batch['summary'].shape[0]  # 46
bat_dict = {}
for i in range(num_cells):  # i: 第几个电池
    print(f'num {i+1} cell is reading')
    cl = f[batch['cycle_life'][i, 0]][:][0][0]  # h5py 3.几的版本没有.vlaue这个属性（降版本）; [:]替换.value
    # print(cl) # [0][0]是取出二维数组唯一的值
    policy = f[batch['policy_readable'][i, 0]][:].tobytes()[::2].decode()

    '''
    Read Data
    Qd 放电容量, Qc 充电容量，Internal Resistance IR(内阻) 越低越好
    Cell1:1008次充放电
    summary是所有一个电池每次cycle的总结数据（一次cycle总结出一个value）
    '''
    summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())  # mat的行列和py的不一样嘛？
    # a:(3,) b:(4,) -> (7,);  a:(3,5) b:(4,5) -> (7,5)   np.hstack((a,b))水平堆栈 == torch.cat((a,b), dim=0)
    # np.hstack 好像只是把列表变为数组
    summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
    summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
    summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
    summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
    summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
    summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
    summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())

    summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':summary_TA,
               'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,'cycle': summary_CY}

    '''
    读取详细的所有cycles的数据
    '''
    cycles = f[batch['cycles'][i, 0]]  # All cycles data
    cycle_dict = {}
    for j in range(len(summary_CY)):  # total cycle num, j : 第几个cycle
        I = np.hstack((f[cycles['I'][j, 0]][:]))
        Qc = np.hstack((f[cycles['Qc'][j, 0]][:]))
        Qd = np.hstack((f[cycles['Qd'][j, 0]][:]))
        Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][:]))
        T = np.hstack((f[cycles['T'][j, 0]][:]))
        Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][:]))
        V = np.hstack((f[cycles['V'][j, 0]][:]))
        dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][:]))
        t = np.hstack((f[cycles['t'][j, 0]][:]))
        cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
        cycle_dict[str(j)] = cd

    cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
    key = 'b3c' + str(i)
    bat_dict[key] = cell_dict
    print()

with open('batch3.pkl', 'wb') as fp:
    pickle.dump(bat_dict, fp)

print(bat_dict.keys())
plt.figure(3)
plt.plot(bat_dict['b3c43']['summary']['cycle'], bat_dict['b3c43']['summary']['QD'])
plt.figure(4)
plt.plot([i for i in range(len(bat_dict['b3c43']['cycles']['10']['V']))], bat_dict['b3c43']['cycles']['10']['V'])
print()
plt.show()
