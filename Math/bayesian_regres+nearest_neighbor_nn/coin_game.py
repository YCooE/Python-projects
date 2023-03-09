import numpy
numpy.random.seed(0)
from scipy.stats import beta
import matplotlib.pyplot as plt

def generate_plots(prior_alpha, prior_beta, title, N=1000, mod_plot=100, heads_prob=0.7):

    plt.figure()

    # plot prior
    x = numpy.linspace(0, 1, 500)
    plt.plot(x, beta.pdf(x, prior_alpha, prior_beta),'b--', lw=2, alpha=1.0, label='prior')

    # draw samples and plot posteriors
    y_i = 0

    for i in range(1,N+1):

        # draw sample
        y = numpy.random.binomial(1, heads_prob, 1)
        y_i += y

        if i % mod_plot == 0:    

            # plot posterior
            delta = y_i + prior_alpha
            gamma = i - y_i + prior_beta

            mean = delta / (delta+gamma)
            alpha = 1.0 * (i/N) 
            beta_values = beta.pdf(x, delta, gamma)
            plt.plot(x, beta_values ,'r-', lw=1, alpha=alpha, label='%i-th posterior: %f' % (i, mean))
    plt.title(title)
    plt.legend()

# 1.
generate_plots(1,1,"alpha=1, beta=1")
generate_plots(100,100,"alpha=100, beta=100")

# 2. 
generate_plots(3,1,"alpha=3, beta=1")

plt.show()

