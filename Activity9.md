# Week 9: Activity Key


### Last Week’s Recap

- Gaussian Process Intro
- Bayesian inference with `stan`
- Correlation functions

### This week

- GPs in 2D
- Bayesian inference with `stan`

------------------------------------------------------------------------

### Geostatistical Data

At last, we will look at simulated 2-d “spatial” data.

##### 1. Create Sampling Locations

##### 2. Calculate Distances

##### 3. Define Covariance Function and Set Parameters

Use exponential covariance with no nugget:

##### 4. Sample realization of the process

- This requires a distributional assumption, we will use the Gaussian
  distribution

- What about the rest of the locations on the map?

##### 5. Vizualize Spatial Process

Start with a coarse grid and them move to a finer grid

Now we can look at more sampling locations

How does the spatial process change with:

- another draw with same parameters?
- a different value of $\phi$
- a different value of $\sigma^2$

------------------------------------------------------------------------

### Visual Overview of Bayesian Inference

Using some Bridger Bowl weather data we will provide a visual overview
of Bayesian Inference. *The goal will be to model the average winter
high temperature at the base of Bridger Bowl.*

1.  Prior Specification

- First sketch a prior distribution that encapsulates your belief about
  what you believe the average high temperature would be. *Note this
  should obey law of total probability*

- Next we generally need to parameterize (perhaps approximately) this
  belief with some sort of probability distribution.

2.  Specify the sampling distribution for the data or perhaps in more
    familiar language, state the likelihood for the statistical model

- We will assume that the temperature readings are continuous (or
  “nearly continuous”)

3.  Posterior Inference

- Grab some weather data from Bridger Bowl (roughly the first half of
  January 2021)

``` r
temp <- c(26, 45, 44, 36, 22, 25, 31, 31, 37, 34, 35, 37,32, 31)
```

- Any concerns about using this data to inform our research question?

- Using classical inference, how would you estimate $\mu$.

- *With Bayesian inference, our posterior belief is based on the data
  **and** our prior belief. Note this can be a blessing or a curse.*

- Formally, we have a distribution for the maximum temperature (a
  posterior distribution):
  $p(\mu|x) = \int p(x|\mu,\sigma) \times p(\mu)p(\sigma) /p(\mu)d\sigma$,
  note solving this is not trivial and isn’t something we will handle in
  this class.

- Luckily, there is an elegant computational procedure that will allow
  us to approximate $p(\mu|x)$ by taking samples from the distribution.
  *This is, of course, MCMC.*

STAN code for this situation can be written as below. Note that the
prior values are hard coded, these could also be passed in as arguments
to the model.

    data {
      int<lower=0> N;
      vector[N] y;
    }


    parameters {
      real mu;
      real<lower=0> sigma;
    }


    model {
      y ~ normal(mu, sigma);
      mu ~ normal(20, 10);
    }

#### Multivariate Normal Distribution

Next we will segue from standard linear models to analyzing correlated
data.

First we will start with the a bivariate normal distribution: y ~
N(theta,sigma), where theta is a mean vector and sigma = sigmasq \* I is
a covariance matrix.

To provide a motivating context, not consider jointly estimating the
temperature at Bridger Bowl *and* Big Sky Resort.

##### 1. Simulate independent bivariate normal

Simulate a set of temperature values from each location, where the
temperature values are independent (sigma = sigmasq \* I)

``` r
library(mnormt)
n <- 100
theta <- c(15,25)
sigma <- diag(2) * 100
fake_temperatures <- rmnorm(n, theta , sigma)
```

Then create a few graphs to show marginal distribution of temperature as
well as how the temperatures evolve in time.

``` r
library(reshape2)
melt(fake_temperatures, value.name = 'temp') %>% 
  rename(location = Var2) %>%
  mutate(location = factor(location)) %>% 
  ggplot(aes(x =temp, fill = location)) + 
  geom_histogram() +
  facet_wrap(.~location) + theme_bw() 
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](Activity9_files/figure-commonmark/unnamed-chunk-3-1.png)

``` r
melt(fake_temperatures, value.name = 'temp') %>% 
  rename(location = Var2, day = Var1) %>%
  mutate(location = factor(location)) %>%
  ggplot(aes(y =temp, x = day, color = location )) + 
  geom_line() + theme_bw() + xlim(0,30) + 
  ggtitle('First 30 observations of independent response')
```

    Warning: Removed 140 rows containing missing values or values outside the scale range
    (`geom_line()`).

![](Activity9_files/figure-commonmark/unnamed-chunk-4-1.png)

##### 2. Simulate correlated bivariate normal

Simulate a set of temperature values from each location, where the
temperature values are not independent (sigma = sigmasq \* H), where H
is a correlation matrix. (Note there are some constraints we will
discuss later)

##### 3. STAN code for bivariate normal

Stan code that will allow you to estimate theta and sigma (including H)

    data {
      int<lower=0> p;
      int<lower=0> N;
      matrix[N,p] y;
    }

    parameters {
      vector[p] theta;
      corr_matrix[p] H;
      real<lower = 0> sigma;
    }

    model {
      for(i in 1:N){
        y[i,:] ~ multi_normal(theta, sigma*H);
      }
    }

##### 4. Use STAN to estimate bivariate normal parameters

Use your stan code to estimate theta and sigma (including H and sigmasq)

##### 5. Final Thoughts About Correlation

In many statistical models there is an assumption about independence.
When independence is violated, uncertainty is under estimated and in
incorrect inferences can be made.

While lack of independence often has a negative connotation, in spatial
statistics we can actually exploit correlation. For instance, by knowing
the temperature at the weather station at Bozeman High School or Bridger
Bowl, we can estimate temperature at other locations.
