library(greta) # prints a bunch or annoying deprecation warnings, but no worry
library(readr)
library(lubridate)
library(tidyverse)

fluvius_data <- read_csv("data/fluvius_data.csv")

y <- fluvius_data$`SSC (mg/L)` # response data
X <- scale(fluvious_data[, 9:11]) #
i <- fluvius_data$region
j <- fluvius_data$site_no
is_leap_year <- lubridate::leap_year(fluvius_data$`Date-Time`) # necessary for harmonic function

# julian_to_harmonic <- function(t, c, days_in_year) {
#   b <- 2*pi / days_in_year
#   sin(b*(t + c))
# }

day_means <- fluvius_data %>% 
  filter() %>% 
  mutate(doy = yday(`Date-Time`)) %>% 
  group_by(doy) %>% 
  summarize(mean(`SSC (mg/L)`))

plot(density(day_means[, 2]))
