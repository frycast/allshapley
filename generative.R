path <- paste0("D:/CloudDrive/OneDrive/Home/",
               "PhD/Projects/Shapley/Nonlinear_dependence/",
               "sunnies/R/")
# path <- paste0("C:/Users/danie/OneDrive/Home/",
#                "PhD/Projects/Shapley/Nonlinear_dependence/",
#                "sunnies/R/")
source(paste0(path,"shapley_helpers.R"))
source(paste0(path,"utility_functions.R"))

## Sample size and initial data
n <- 100
y <- runif(n)

# Target characteristic function
S <- list(1,2,3,c(1,2),c(1,3),c(2,3),c(1,2,3))
target <- function(s) {
  suppressWarnings({
  if (all(s==1))        {return(0)}
  if (all(s==2))        {return(0.7)}
  if (all(s==3))        {return(0.7)}
  if (all(s==c(1,2)))   {return(1)}
  if (all(s==c(1,3)))   {return(1)}
  if (all(s==c(2,3)))   {return(0.7)}
  if (all(s==c(1,2,3))) {return(1)}
  stop(paste0("failed on ",s))
  })
}

u123 <- 0.8
u13  <- 0.8
u12  <- 0.8
u23  <- 0.5
u3   <- 0.5
u2   <- 0.5
u1   <- 0
phi2 <- (u123 - u13 + (u23 - u3 + u12 - u1)/2 + u2)/3; phi2
phi3 <- (u123 - u12 + (u23 - u2 + u13 - u1)/2 + u3)/3; phi3
phi1 <- (u123 - u23 + (u12 - u2 + u13 - u3)/2 + u1)/3; phi1

# # minimize
# u1 + abs(u23 - u2) + abs(u23 - u3)
# 
# # maximize
# u12 + u13 + u123 + (u12 - u23) + (u12 - u2) + (u13 - u3)
# 
# # maximise
# phi2 <- (u123 - u13 + (u23 - u3 + u12 - u1)/2 + u2)/3
# phi3 <- (u123 - u12 + (u23 - u2 + u13 - u1)/2 + u3)/3
# u123 + u12 + u13
# # becomes
# 10*u123/6 + 5*u13/6 + 5*u12/6 + 2*u23/3 + u3/6 + u2/6 - u1/3
# 
# # + minimise
# phi1 <- (u123 - u23 + (u12 - u2 + u13 - u3)/2 + u1)/3
# u23 + u2 + u3
# 
# ## Finally:
# # becomes
# 8*u123/6 + 4*u13/6 + 4*u12/6 - 4*u3/6 - 4*u2/6 - 4*u1/6
# ## Loss function
# loss <- function(y,X,S,U=DC) {
#   u123 <- U(y,X[,c(1,2,3)])
#   u13  <- U(y,X[,c(1,3)])  
#   u12  <- U(y,X[,c(1,2)])  
#   u3   <- U(y,X[,c(3)])  
#   u2   <- U(y,X[,c(2)])  
#   u1   <- U(y,X[,c(1)]) 
#   -(8*u123/6 + 4*u13/6 + 4*u12/6 - 4*u3/6 - 4*u2/6 - 4*u1/6)
# }

# loss <- function(y,X,S,U=DC) {
#   u123 <- U(y,X[,c(1,2,3)])
#   u23  <- U(y,X[,c(2,3)])
#   u13  <- U(y,X[,c(1,3)])
#   u12  <- U(y,X[,c(1,2)])
#   u3   <- U(y,X[,c(3)])
#   u2   <- U(y,X[,c(2)])
#   u1   <- U(y,X[,c(1)])
#   (u123 - u23 + (u12 - u2 + u13 - u3)/2 + u1)/3 +
#     - (u123 - u13 + (u23 - u3 + u12 - u1)/2 + u2)/3 +
#     - (u123 - u12 + (u23 - u2 + u13 - u1)/2 + u3)/3 +
#     - (u123 + u12 + u13) +
#     u23 + u2 + u3
# }


# loss <- function(y,X,S,U=DC) {
#   u123 <- U(y,X[,c(1,2,3)])
#   u23  <- U(y,X[,c(2,3)])
#   u13  <- U(y,X[,c(1,3)])
#   u12  <- U(y,X[,c(1,2)])
#   u3   <- U(y,X[,c(3)])
#   u2   <- U(y,X[,c(2)])
#   u1   <- U(y,X[,c(1)])
#   l123 <- abs(u123 - 1) 
#   l13 <- abs(u13 - 1) 
#   l12 <- abs(u12 - 1)
#   l23 <- abs(u23 - 0.7) 
#   l3 <- abs(u3 - 0.7)
#   l2 <- abs(u2 - 0.7) 
#   l1 <- abs(u1)
#   eps <- c(0.5,0.1,0.05,0.01)
#   alpha <- 4
#   l <- c(l123, l13, l12, l23, l3, l2, l1,0)
#   loss_out <- sum(l[l<eps[1]]) + 
#     2*sum(l[l<eps[2]]) + 
#     3*sum(l[l<eps[3]]) + 
#     4*sum(l[l<eps[4]])
#   return(alpha - loss_out)
# }

loss <- function(y,X,S,U=DC) {
  u123 <- U(y,X[,c(1,2,3)])
  u23  <- U(y,X[,c(2,3)])
  u13  <- U(y,X[,c(1,3)])
  u12  <- U(y,X[,c(1,2)])
  u3   <- U(y,X[,c(3)])
  u2   <- U(y,X[,c(2)])
  u1   <- U(y,X[,c(1)])
  abs(u123 - 0.8) + abs(u13 - 0.8) + abs(u12 - 0.8) + 
    abs(u23 - 0.5) + abs(u3 - 0.5) + abs(u2 - 0.5) + abs(u1) +
    0 #+ 2*abs(u123 - u12) + 2*abs(u123 - u13) - u12 - u13 + u23
} 

# ## Loss function
# loss <- function(y, X, S, U = DC) {
#   sum(unlist(lapply(S, function(s){abs(U(y,X[,s]) - target(s))})))
# }

## Grid search add row
grid_best_row <- function(y,X,S,vals = c(-1,0,1)) {
  candidates <- expand.grid(vals,vals,vals)
  losses <- vector(length = nrow(candidates), mode = "numeric")
  colnames(X) <- colnames(candidates)
  for (i in 1:nrow(candidates)) {
    X2 <- rbind(X,candidates[i,])
    losses[i] <- loss(y[1:nrow(X2)],X2,S,U=R2)
  }
  return(rbind(X, candidates[which.min(losses),]))
}

## The grid add row approach
X <- matrix(c(0,0,0), nrow = 1, ncol = 3)
losses <- vector(mode = "numeric", length = n)
for (i in 1:100) {
  X <- grid_best_row(y,X,S)
  losses[i] <- loss(y[1:nrow(X)],X,S)
}

## Replace m% approach
replace_m_percent <- function(Z,S,m=20,N=20,U=R2) {
  all_Z2 <- list()
  all_Z2[[1]] <- Z
  n <- nrow(Z)
  rn <- floor(m/100*n)
  current_loss <- loss(Z[,1],Z[,-1],S,U=R2)
  losses <- rep(current_loss, N)
  for (i in 2:(N+1)) {
    r <- sample(1:n, rn)
    all_Z2[[i]] <- Z
    all_Z2[[i]][r,] <- matrix(runif(rn*4), ncol = 4, nrow = rn)
    losses[i] <- loss(all_Z2[[i]][,1],all_Z2[[i]][,-1],S,U=R2)
    if (losses[1] > losses[i]) break
  }
  
  return(list(Z = all_Z2[[which.min(losses)]],
              loss = losses[which.min(losses)]))
}


## Replace m% linear approach
replace_m_percent_linear <- function(Z,S,m=20,N=20, U=DC) {
  all_Z2 <- list()
  all_Z2[[1]] <- Z
  n <- nrow(Z)
  rn <- floor(m/100*n)
  current_loss <- loss(Z[,1],Z[,-1],S,U=U)
  losses <- rep(current_loss, N)
  rand_X <- matrix(runif(rn*3), ncol = 3, nrow = rn)
  for (i in 2:(N+1)) {
    r <- sample(1:n, rn)
    all_Z2[[i]] <- Z
    rand_B <- runif(3,-1,1)
    all_Z2[[i]][r,] <- cbind( rand_X %*% rand_B, rand_X)
    losses[i] <- loss(all_Z2[[i]][,1],all_Z2[[i]][,-1],S,U=U)
    if (losses[1] > losses[i]) break
  }
  
  return(list(Z = all_Z2[[which.min(losses)]],
              loss = losses[which.min(losses)]))
}

N <- 200
n <- 100
Z <- matrix(runif(n*4), ncol = 4, nrow = n)
keep_losses <- c()

losses <- vector(length = N, mode = "numeric")
for (i in 1:N) {
  result <- replace_m_percent_linear(Z,S,m=10,U=R2)
  Z <- result$Z
  losses[i] <- result$loss
}

keep_losses <- c(keep_losses, losses)
plot(keep_losses, type = 'b')

shapley(Z[,1],Z[,-1], utility = DC)

# See the characteristic function
CF <- estimate_CF(Z[,-1], utility = R2, y = Z[,1])
CF(c(1,2,3)) #1
CF(c(2,3))   #0.7
CF(c(1,2))   #1
CF(c(1,3))   #1
CF(1)        #0
CF(2)        #0.7
CF(3)        #0.7


###########################################################################
# MANUAL NN ATTEMPT -------------------------------------------------------
###########################################################################

### Neural network code was adapted from:
# https://towardsdatascience.com/build-your-own-neural-network-classifier-in-r-b7f1f183261d

# Target dataset sample size: n
# Target dataset number of features: d
# Input layer: n*d nodes 
# Input observations: one random dataset per observation
# Input isn't a totally connected layer though: only 
# Second last layer: a vector y of n nodes
# Activation (last layer), uses input X as well, and outputs the CF
# Loss is CF MSE.

# u123 <- 0.8
# u13  <- 0.8
# u12  <- 0.8
# u23  <- 0.5
# u3   <- 0.5
# u2   <- 0.5
# u1   <- 0
# target <- matrix(rep(c(u123,u13,u12,u23,u3,u2,u1), N), nrow = N, byrow = T)

n <- 100 # target data set sample size
d <- 3 # target number of features

# input data
X <- matrix(runif(n*d), nrow = n, ncol = d)

# The characteristic function loss
loss_func <- function(y,X,U=DC) {
  u123 <- U(y,X[,c(1,2,3)])
  u23  <- U(y,X[,c(2,3)])
  u13  <- U(y,X[,c(1,3)])
  u12  <- U(y,X[,c(1,2)])
  u3   <- U(y,X[,c(3)])
  u2   <- U(y,X[,c(2)])
  u1   <- U(y,X[,c(1)])
  (u123 - 0.8)^2 + (u13 - 0.8)^2 + (u12 - 0.8)^2 + 
    (u23 - 0.5)^2 + (u3 - 0.5)^2 + (u2 - 0.5)^2 + (u1)^2
}


## R squared equation might be easier to differentiate than dcor
# 
# # Equation (2) (taking care of 0-indexing)
# Cn_ = cor(Z)
# Cn(u) = [Cn_[i,j] for i in u .+ 1, j in u .+ 1]
# 
# # Equation (4) (taking care of empty s here)
# R2(s) = (length(s) > 0) ? 1 - det(Cn(vcat(0,s)))/det(Cn(s)) : 0
#

nnet <- function(X, step_size = 0.5, reg = 0.001, h = 10, niteration){
  # get dim of input
  N <- nrow(X) # number of examples
  K <- 1 # number of y (sample size)
  D <- ncol(X) # dimensionality
  
  # initialize parameters randomly
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)
  b <- matrix(0, nrow = 1, ncol = h)
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h)
  b2 <- matrix(0, nrow = 1, ncol = K)
  
  # gradient descent loop to update weight and bias
  for (i in 0:niteration){
    # hidden layer, ReLU activation
    hidden_layer <- pmax(0, X %*% W + matrix(rep(b,N), nrow = N, byrow = T))
    hidden_layer <- matrix(hidden_layer, nrow = N)
    # class score
    y_nodes <- hidden_layer %*% W2 + matrix(rep(b2,N), nrow = N, byrow = T)
    
    # compute the loss: characteristic function loss
    data_loss <- loss_func(y_nodes_mean, X)
    reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2)
    loss <- data_loss + reg_loss
    # check progress
    if (i%%10 == 0 | i == niteration){
      print(paste("iteration", i,': loss', loss))}
    
    # compute the gradient on scores
    
    # THE DIMENSION IS RIGHT BUT THIS IS ONLY A PLACEHOLDER (ZERO GRADIENT)
    # COMPUTING THE GRADIENT IS A PAIN: MOVE TO KERAS / TENSORFLOW
    dscores <- matrix(0.001,nrow=N) #y_nodes  
    
    # backpropagate the gradient to the parameters
    dW2 <- t(hidden_layer)%*%dscores
    db2 <- colSums(dscores)
    # next backprop into hidden layer
    dhidden <- dscores%*%t(W2)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] <- 0
    # finally into W,b
    dW <- t(X)%*%dhidden
    db <- colSums(dhidden)
    
    # add regularization gradient contribution
    dW2 <- dW2 + reg *W2
    dW <- dW + reg *W
    
    # update parameter 
    W <- W-step_size*dW
    b <- b-step_size*db
    W2 <- W2-step_size*dW2
    b2 <- b2-step_size*db2
  }
  return(list(W, b, W2, b2, y_nodes))
}

nnetPred <- function(X, para = list()){
  W <- para[[1]]
  b <- para[[2]]
  W2 <- para[[3]]
  b2 <- para[[4]]
  
  N <- nrow(X)
  hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T)) 
  hidden_layer <- matrix(hidden_layer, nrow = N)
  y_nodes <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)
  
  return(y_nodes)  
}

nnet.model <- nnet(X, step_size = 0.4,reg = 0.0002, h=50, niteration = 1000)
## [1] "iteration 0 : loss 1.38628868932674"
## [1] "iteration 1000 : loss 0.967921639616882"
## [1] "iteration 2000 : loss 0.448881467342854"
## [1] "iteration 3000 : loss 0.293036646147359"
## [1] "iteration 4000 : loss 0.244380009480792"
## [1] "iteration 5000 : loss 0.225211501612035"
## [1] "iteration 6000 : loss 0.218468573259166"
predicted_class <- nnetPred(X, nnet.model)
print(paste('training accuracy:',mean(predicted_class == (y))))



###########################################################################
# KERAS NN ATTEMPT --------------------------------------------------------
###########################################################################

library(tensorflow)

x <- tf$ones(shape(2, 2))

with(tf$GradientTape() %as% t, {
  t$watch(x)
  y <- tf$reduce_sum(x)
  z <- tf$multiply(y, y)
})

# Derivative of z with respect to the original input tensor x
dz_dx <- t$gradient(z, x)
dz_dx




