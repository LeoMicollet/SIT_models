library(IBMPopSim)
library(dplyr)

# Climate data
meteo_df <- read.csv("../Data/meteo_tetiaroa_resampled.csv")
meteo_df <- meteo_df[complete.cases(meteo_df[, c("precip","precip1","precip2",
                                                   "precip4","UM","TM",
                                                   "date","time")]), ]

# Build a numeric time axis (one row = one day)
time_data    <- seq(0, nrow(meteo_df) - 1, by = 1)
precip2_data <- meteo_df$precip2
temp_data    <- meteo_df$TM        # degrees C

T_end <- 365 # simulation end (days)


# Parameters 
birth         <- 10 / 64
n_egg         <- 64L
egg_death     <- 0.046
larva_death   <- 0.05
pupa_death    <- 0.05
female_death  <- 0.046
male_death    <- 0.139
sterile_death <- male_death * 1.2   # = 0.1668
c_comp        <- 0.001
transi_el     <- 0.79
transi_lp     <- 0.125
transi_pa     <- 0.125

params_base <- list(
  egg_death    = egg_death,
  larva_death  = larva_death,
  pupa_death   = pupa_death,
  female_death = female_death,
  male_death   = male_death,
  sterile_death= sterile_death,
  birth        = birth,
  n_egg        = n_egg,
  transi_el    = transi_el,
  transi_lp    = transi_lp,
  transi_pa    = transi_pa,
  c_comp       = c_comp
)

lut_dt    <- 0.5
lut_times <- seq(0, ceiling(T_end), by = lut_dt)
lut_n     <- as.integer(length(lut_times))

lut_P  <- approx(time_data, precip2_data, xout = lut_times, rule = 2)$y

LUT_INTERP <- "
  auto lut_at = [](const arma::vec& v, double t,
                   double dt, int n) -> double {
    double id = t / dt;
    int    i  = (int)id;
    if (i < 0)    i = 0;
    if (i >= n-1) i = n-2;
    double f = id - i;
    return v[i]*(1.0-f) + v[i+1]*f;
  };
"


# First sim the sterile population

RELEASE_SIZE  <- 10000L
RELEASE_START <- 250.0
N_RELEASES    <- 20L
RELEASE_EVERY <- 7.0

release_times <- RELEASE_START + (0:(N_RELEASES - 1)) * RELEASE_EVERY
release_times <- release_times[release_times <= T_end]

pop_sterile_init <- population(
  data.frame(birth = -1, death = -1)
)

ev_sterile_death <- mk_event_individual(
  type           = "death",
  name           = "sterile_death",
  intensity_code = "result = sterile_death;"
)

model_sterile <- mk_model(
  characteristics = get_characteristics(pop_sterile_init),
  events          = list(ev_sterile_death),
  parameters      = list(sterile_death = sterile_death)
)

cat("=== STEP 1: Sterile population simulation ===\n")

breakpoints   <- c(release_times, T_end)
pop_s_cur     <- pop_sterile_init
t_s_cur       <- 0.0
sterile_times <- numeric(0)
sterile_N     <- numeric(0)


# Seed: one pre-dead dummy so IBMPopSim can infer the schema
pop_s_cur <- data.frame(birth = -1.0, death = -1.0)

run_sterile_seg <- function(pop, dt) {
  # Reset alive individuals' birth to 0 (age-irrelevant since rate is constant)
  # and strip any extra columns popsim may have added previously.
  alive_mask        <- is.na(pop$death)
  pop$birth[alive_mask] <- 0.0
  pop_clean <- population(pop[, c("birth", "death")])
  sim <- popsim(
    model              = model_sterile,
    initial_population = pop_clean,
    events_bounds      = c(sterile_death = sterile_death),
    parameters         = list(sterile_death = sterile_death),
    time               = dt,
    multithreading     = FALSE
  )
  res <- sim$population
  # Return a clean 2-column frame: alive individuals have death=NA
  data.frame(
    birth = ifelse(is.na(res$death),  0.0,        -1.0),
    death = ifelse(is.na(res$death),  NA_real_,   -1.0)
  )
}

for (i in seq_along(breakpoints)) {
  t_next <- breakpoints[i]
  if (t_next <= t_s_cur) next

  # ---- Daily sub-segments between t_s_cur and t_next ----
  daily_grid <- seq(t_s_cur, t_next, by = 1.0)
  if (tail(daily_grid, 1) < t_next) daily_grid <- c(daily_grid, t_next)

  for (j in seq_along(daily_grid[-1])) {
    dt_sub    <- daily_grid[j + 1] - daily_grid[j]
    pop_s_cur <- run_sterile_seg(pop_s_cur, dt_sub)
    t_s_cur   <- daily_grid[j + 1]

    n_alive       <- sum(is.na(pop_s_cur$death))
    sterile_times <- c(sterile_times, t_s_cur)
    sterile_N     <- c(sterile_N,     n_alive)
  }

  cat(sprintf("  t=%6.1f  Sterile alive (before release): %d\n",
              t_s_cur, tail(sterile_N, 1)))

  # ---- Inject release batch at this breakpoint ----
  if (i < length(breakpoints)) {
    new_block <- data.frame(
      birth = rep(0.0,      RELEASE_SIZE),
      death = rep(NA_real_, RELEASE_SIZE)
    )
    pop_s_cur <- rbind(pop_s_cur, new_block)

    # Record the spike IMMEDIATELY after the release (same time + epsilon)
    n_after   <- sum(is.na(pop_s_cur$death))
    sterile_times <- c(sterile_times, t_s_cur + 1e-6)
    sterile_N     <- c(sterile_N,     n_after)

    cat(sprintf("  --> Release %d at t=%.1f: +%d  (total alive: %d)\n",
                i, t_s_cur, RELEASE_SIZE, n_after))
  }
}

# Build N_s interpolation function and LUT.
# method="constant" gives piecewise-constant interpolation (correct for a
# population that only changes at discrete events).
N_sterile_fun <- approxfun(sterile_times, sterile_N,
                            method = "constant", rule = 2)
# Evaluate N_sterile_fun on the LUT grid to get a plain numeric vector
lut_Ns <- N_sterile_fun(lut_times)

cat("Step 1 complete.\n\n")

plot(sterile_times, sterile_N,
     type = "l", col = "purple",
     xlab = "Time (days)", ylab = "Sterile males alive",
     main = "Step 1 — Sterile population over time")

# Mark each release with a vertical dashed line
abline(v = release_times, col = "gray50", lty = 2)

# Add points at each recorded time
points(sterile_times, sterile_N, pch = 19, cex = 0.5, col = "purple")


# Simu pop 
init_counts <- c(Egg=500L, Larva=200L, Pupa=100L,
                 FemaleV=80L, FemaleM=50L, FemaleS=10L, Male=60L)
stages_vec <- c(rep(0L, init_counts["Egg"]),
                rep(1L, init_counts["Larva"]),
                rep(2L, init_counts["Pupa"]),
                rep(3L, init_counts["FemaleV"]),
                rep(4L, init_counts["FemaleM"]),
                rep(5L, init_counts["FemaleS"]),
                rep(6L, init_counts["Male"]))

pop_full_init <- population(data.frame(
  birth = rep(0.0, length(stages_vec)),
  death = rep(NA_real_, length(stages_vec)),
  stage = as.integer(stages_vec)
))

# N_wild is the only parameter updated per-segment; all others are constant.
params_full <- c(
  params_base,
  list(
    lut_P  = lut_P,
    lut_Ns = lut_Ns,
    lut_dt = lut_dt,
    lut_n  = lut_n,
    N_wild = as.double(init_counts["Male"])
  )
)

# ---- Events ----

# Death — one event, intensity branches on stage
ev_death <- mk_event_individual(
  type = "death",
  name = "death",
  intensity_code = '
    int s = I.stage;
    if      (s == 0) result = egg_death;
    else if (s == 1) result = larva_death;  // base rate; +competition below
    else if (s == 2) result = pupa_death;
    else if (s == 3 || s == 4 || s == 5) result = female_death;
    else if (s == 6) result = male_death;
    else             result = 0.0;
  '
)

# Larva competition death — quadratic interaction
# W(I,J) non-zero only when both are larvae.
# Total rate for larva I = sum_J W(I,J) = comp_value * N_larvae  (matches Python)
ev_larva_comp <- mk_event_interaction(
  type = "death",
  name = "larva_comp",
  interaction_code = paste0(LUT_INTERP, '
    if (I.stage == 1 && J.stage == 1) {
      double precip = lut_at(lut_P, t, lut_dt, lut_n);
      // competition1(K0=1/c, Kh=1/(c*50), precip) = 1 / (K0 + Kh*precip)
      double K0 = 1.0 / c_comp;
      double Kh = 1.0 / (c_comp * 50.0);
      result = 1.0 / (K0 + Kh * precip);
    } else {
      result = 0.0;
    }
  ')
)

# Stage transitions (swap events)
ev_egg_larva <- mk_event_individual(
  type = "swap", name = "egg_to_larva",
  intensity_code = "result = (I.stage == 0) ? transi_el : 0.0;",
  kernel_code    = "I.stage = 1;"
)

ev_larva_pupa <- mk_event_individual(
  type = "swap", name = "larva_to_pupa",
  intensity_code = "result = (I.stage == 1) ? transi_lp : 0.0;",
  kernel_code    = "I.stage = 2;"
)

ev_pupa_male <- mk_event_individual(
  type = "swap", name = "pupa_to_male",
  intensity_code = "result = (I.stage == 2) ? (0.5 * transi_pa) : 0.0;",
  kernel_code    = "I.stage = 6;"
)

ev_pupa_femaleV <- mk_event_individual(
  type = "swap", name = "pupa_to_femaleV",
  intensity_code = "result = (I.stage == 2) ? (0.5 * transi_pa) : 0.0;",
  kernel_code    = "I.stage = 3;"
)

# Mating: Virgin -> FemaleM  (mated by wild male)
# rate = allee(N_wild, N_sterile) * prob_M
#      = allee(Nw, Ns) * Nw / (Nw + Ns)
ev_mating_wild <- mk_event_individual(
  type = "swap", name = "mating_wild",
  intensity_code = paste0(LUT_INTERP, '
    if (I.stage != 3) { result = 0.0; }
    else {
      double Nw   = N_wild;
      double Ns   = lut_at(lut_Ns, t, lut_dt, lut_n);
      double Ntot = Nw + Ns;
      if (Ntot <= 0.0 || Nw <= 0.0) { result = 0.0; }
      else {
        double num_a = 0.1*Nw + 0.01*Ns;
        double allee = num_a / (0.5 + num_a);
        result = allee * (Nw / Ntot);
      }
    }
  '),
  kernel_code = "I.stage = 4;"
)

# Mating: Virgin -> FemaleS  (mated by sterile male)
# rate = allee(N_wild, N_sterile) * prob_Ms
#      = allee(Nw, Ns) * Ns / (Nw + Ns)
ev_mating_sterile <- mk_event_individual(
  type = "swap", name = "mating_sterile",
  intensity_code = paste0(LUT_INTERP, '
    if (I.stage != 3) { result = 0.0; }
    else {
      double Nw   = N_wild;
      double Ns   = lut_at(lut_Ns, t, lut_dt, lut_n);
      double Ntot = Nw + Ns;
      if (Ntot <= 0.0 || Ns <= 0.0) { result = 0.0; }
      else {
        double num_a = 0.1*Nw + 0.01*Ns;
        double allee = num_a / (0.5 + num_a);
        result = allee * (Ns / Ntot);
      }
    }
  '),
  kernel_code = "I.stage = 5;"
)

ev_birth <- mk_event_individual(
  type = "birth", name = "egg_birth",
  intensity_code = "result = (I.stage == 4) ? (birth * (double)n_egg) : 0.0;",
  kernel_code    = "newI.stage = 0;"
)

# ---- Build model ----
model_full <- mk_model(
  characteristics   = get_characteristics(pop_full_init),
  events            = list(ev_death,
                           ev_larva_comp,
                           ev_egg_larva,
                           ev_larva_pupa,
                           ev_pupa_male,
                           ev_pupa_femaleV,
                           ev_mating_wild,
                           ev_mating_sterile,
                           ev_birth),
  parameters        = params_full
)

# Thinning upper bound 
comp_bound <-c_comp   # = 100

events_bounds_full <- c(
  death           = max(egg_death, larva_death, pupa_death,
                        female_death, male_death),   # = 0.139
  larva_comp      = comp_bound,                       # = 100
  egg_to_larva    = transi_el,                        # = 0.79
  larva_to_pupa   = transi_lp,                        # = 0.125
  pupa_to_male    = 0.5 * transi_pa,                  # = 0.0625
  pupa_to_femaleV = 0.5 * transi_pa,                  # = 0.0625
  mating_wild     = 1.0,    # allee() < 1 always
  mating_sterile  = 1.0,
  egg_birth       = birth * n_egg                     # = 10
)

# Simiulation with daily update
segment_dt <- 1.0
t_segments <- seq(0, T_end, by = segment_dt)

count_pop <- function(pop_df, t) {
  alive <- pop_df[is.na(pop_df$death), ]
  data.frame(
    t       = t,
    Egg     = sum(alive$stage == 0L),
    Larva   = sum(alive$stage == 1L),
    Pupa    = sum(alive$stage == 2L),
    FemaleV = sum(alive$stage == 3L),
    FemaleM = sum(alive$stage == 4L),
    FemaleS = sum(alive$stage == 5L),
    Male    = sum(alive$stage == 6L),
    Sterile = as.integer(round(approx(lut_times, lut_Ns, xout = t, rule = 2)$y))
  )
}

count_list <- vector("list", length(t_segments))
pop_cur    <- pop_full_init
t_cur      <- 0.0
params_cur <- params_full


cat("=== STEP 2: Full 7-compartment simulation ===\n")

for (i in seq_along(t_segments[-1])) {
  t_next <- t_segments[i + 1]

  alive_now  <- pop_cur[is.na(pop_cur$death), ]
  N_wild_now <- sum(alive_now$stage == 6L)
  params_cur$N_wild <- as.double(N_wild_now)

  if (nrow(alive_now) == 0L) {
    cat(sprintf("Population extinct at t=%.2f\n", t_cur))
    break
  }

  sim <- tryCatch(
    popsim(
      model              = model_full,
      initial_population = pop_cur,
      events_bounds      = events_bounds_full,
      parameters         = params_cur,
      time               = t_next - t_cur,
      multithreading     = FALSE
    ),
    error = function(e) {
      cat(sprintf("Error at t=%.2f: %s\n", t_cur, e$message)); NULL
    }
  )
  if (is.null(sim)) break

  pop_cur <- sim$population
  t_cur   <- t_next
  cnt     <- count_pop(pop_cur, t_cur)
  count_list[[i]] <- cnt

  if (i %% 50 == 0)
    cat(sprintf("t=%6.1f | E=%4d L=%4d P=%4d Fv=%3d Fm=%3d Fs=%3d M=%3d Ms=%4d\n",
                t_cur, cnt$Egg, cnt$Larva, cnt$Pupa,
                cnt$FemaleV, cnt$FemaleM, cnt$FemaleS,
                cnt$Male, cnt$Sterile))
}

count_history <- do.call(rbind, Filter(Negate(is.null), count_list))
cat("Simulation complete.\n")

# Plot
if (!is.null(count_history) && nrow(count_history) > 0) {
  old_par <- par(mfrow = c(2, 1), mar = c(4, 4, 2, 1))

  adult_ylim <- c(0, max(count_history[, c("FemaleV","FemaleM","FemaleS",
                                            "Male","Sterile")]))
  plot(count_history$t, count_history$Male, type="l", col="blue",
       ylim = adult_ylim,
       xlab = "Time (days)", ylab = "Count", main = "Adults")
  lines(count_history$t, count_history$FemaleV, col = "pink")
  lines(count_history$t, count_history$FemaleM, col = "red")
  lines(count_history$t, count_history$FemaleS, col = "orange")
  lines(count_history$t, count_history$Sterile,  col = "purple", lty = 2)
  legend("topright", bty="n", cex=0.8,
         legend = c("Male (wild)","FemaleV","FemaleM","FemaleS","Sterile (LUT)"),
         col    = c("blue","pink","red","orange","purple"),
         lty    = c(1,1,1,1,2))

  juv_ylim <- c(0, max(count_history[, c("Egg","Larva","Pupa")]))
  plot(count_history$t, count_history$Egg, type="l", col="green4",
       ylim = juv_ylim,
       xlab = "Time (days)", ylab = "Count", main = "Juveniles")
  lines(count_history$t, count_history$Larva, col = "darkgreen")
  lines(count_history$t, count_history$Pupa,  col = "brown")
  legend("topright", bty="n", cex=0.8,
         legend = c("Egg","Larva","Pupa"),
         col    = c("green4","darkgreen","brown"), lty = 1)

  par(old_par)
}



old_par <- par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))

# ── [1,1] Female population ──────────────────────────────────────────────────
ylim_f <- c(0, max(count_history$FemaleV, count_history$FemaleM, count_history$FemaleS))
plot(count_history$t, count_history$FemaleV, type = "s", col = "blue",  ylim = ylim_f,
     xlab = "Time", ylab = "Population Size", main = "Female Population")
lines(count_history$t, count_history$FemaleM, type = "s", col = "red")
lines(count_history$t, count_history$FemaleS, type = "s", col = "grey")
legend("topright", bty = "n", legend = c("F0","Ff","Fs"),
       col = c("blue","red","grey"), lty = 1)

# ── [1,2] Sterile male population ────────────────────────────────────────────
ylim_s <- c(0, max(count_history$Sterile, 1))
plot(count_history$t, count_history$Sterile, type = "s", col = "orange", ylim = ylim_s,
     xlab = "Time", ylab = "Sterile Male Population Size",
     main = "Sterile Male Population")
legend("topright", bty = "n", legend = "Sterile Male", col = "orange", lty = 1)

# ── [2,1] Larva population ───────────────────────────────────────────────────
ylim_l <- c(0, max(count_history$Larva, 1))
plot(count_history$t, count_history$Larva, type = "s", col = "green4", ylim = ylim_l,
     xlab = "Time", ylab = "Population Size", main = "Larva Population")
legend("topright", bty = "n", legend = "Larva", col = "green4", lty = 1)

# ── [2,2] All females + wild males ───────────────────────────────────────────
total_f <- count_history$FemaleV + count_history$FemaleM + count_history$FemaleS
ylim_m  <- c(0, max(total_f, count_history$Male, 1))
plot(count_history$t, total_f,  type = "s", col = "purple", ylim = ylim_m,
     xlab = "Time", ylab = "Population Size",
     main = "F0+Ff+Fs and fertile female Population")
lines(count_history$t, count_history$Male, type = "s", col = "black")
legend("topright", bty = "n", legend = c("F0+Ff+Fs", "M"),
       col = c("purple", "black"), lty = 1)

par(old_par)
