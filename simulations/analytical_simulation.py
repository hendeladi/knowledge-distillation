import matplotlib.pyplot as plt
from src.statistics import expected_risk_kd, expected_risk_baseline
from src.sets import Region

if __name__ == "__main":
    k_range = list(range(2, 102, 2))
    risk_kd = expected_risk_kd(k_range)
    risk_baseline = expected_risk_baseline(k_range)
    plt.figure()
    plt.plot(k_range, risk_kd, 'r', k_range, risk_baseline, 'b')
    plt.title("kd student expected risk as function of number of examples")
    plt.xlabel("number of training examples")
    plt.ylabel("expected risk")
    plt.legend(["KD", "baseline"])
    # plt.xlim(2, 100)
    # plt.ylim(0.2, 0.4)
    plt.show()
    '''
    x_range = np.linspace(0, 1, 500)
    density2_kd = []
    density4_kd = []
    density10_kd = []
    density40_kd = []
    for x in x_range:
        density2_kd.append(density_kd(x, 2))
        density4_kd.append(density_kd(x, 4))
        density10_kd.append(density_kd(x, 10))
        density40_kd.append(density_kd(x, 40))
    plt.figure()
    plt.plot(x_range, density2_kd, 'b', x_range, density4_kd, 'r', x_range, density10_kd, 'g', x_range, density40_kd, 'purple')
    plt.title("KD probability density function of boundary parameter")
    plt.legend(["k=2", "k=4", "k=10", "k=40"])
    plt.xlabel("boundary parameter")
    plt.ylabel("density")



    x_range = np.linspace(0, 1, 500)
    density2_baseline = []
    density4_baseline = []
    density10_baseline = []
    density40_baseline = []
    for x in x_range:
        density2_baseline.append(density_baseline(x, 2))
        density4_baseline.append(density_baseline(x, 4))
        density10_baseline.append(density_baseline(x, 10))
        density40_baseline.append(density_baseline(x, 40))
    plt.figure()
    plt.plot(x_range, density2_baseline, 'b', x_range, density4_baseline, 'r', x_range, density10_baseline, 'g', x_range, density40_baseline, 'purple')
    plt.title("baseline probability density function of boundary parameter")
    plt.legend(["k=2", "k=4", "k=10", "k=40"])
    plt.xlabel("boundary parameter")
    plt.ylabel("density")




    plt.figure()
    plt.plot(x_range, density10_kd, 'b',x_range, density10_baseline, 'r', x_range, density40_kd, 'g', x_range, density40_baseline, 'purple')
    plt.title("baseline probability density function of boundary parameter")
    plt.legend(["kd 10", "baseline 10", "kd 40", "baseline 40"])
    plt.xlabel("boundary parameter")
    plt.ylabel("density")


    k_range = list(range(2,52,2))
    n = 20000
    delta = 1/n
    x_range = np.linspace(0, 1, n)
    entropy_kd = eval_entropy(density_kd, x_range, k_range)
    entropy_baseline = eval_entropy(density_baseline, x_range, k_range)

    plt.figure()
    plt.plot(k_range, entropy_baseline, 'b', k_range, entropy_kd, 'r')
    plt.title("entropy as function of number of examples")
    plt.xlabel("number of training examples")
    plt.legend(["baseline", "KD"])
    plt.ylabel("entropy")
    plt.show()



    a = eval_entropy(density_kd,x_range,k_range)
    plt.figure()
    plt.plot(k_range, a, 'b')
    plt.title("entropy")
    plt.xlabel("number of training examples")
    plt.ylabel("entropy")
    plt.show()

    eps = 0.0000000001
    n = 50000
    delta = 1/n
    x_range = np.linspace(0, 1, n)
    k_range = [2,10,40]#list(range(2, 62,2))
    entropy = []
    pdf = np.zeros((k_range[-1]+1, len(x_range)))
    for k in k_range:
        integ = 0
        for i, x in enumerate(x_range):
            pdf[k][i] = density_kd(x, k)
            integ += -delta*(pdf[k][i]*np.log2(pdf[k][i] + eps))
            #integ += delta * (pdf[k][i])
        entropy.append(integ)


    plt.figure()
    plt.plot(k_range, entropy, 'b')
    plt.title("entropy a dsfdws")
    plt.xlabel("number of training examples")
    plt.ylabel("entropy")
    plt.show()
    '''



