# https://github.com/jgaeb/measure-mismeasure/blob/main/diabetes.R

set.seed(31371648)
options(tidyverse.quiet = TRUE)

library(groundhog)
groundhog.library(tidyverse, "2023-08-01")

download.file("https://wwwn.cdc.gov/nchs/nhanes/2011-2012/DEMO_G.XPT",
              demo <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2011-2012/DIQ_G.XPT",
              diq <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2011-2012/BMX_G.XPT",
              bmx <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2011-2012/GHB_G.XPT",
              ghb <- tempfile(), mode="wb", quiet = TRUE)
raw_demographics_11_12 <- foreign::read.xport(demo) %>% 
  janitor::clean_names()
raw_survey_responses_11_12 <- foreign::read.xport(diq) %>% 
  janitor::clean_names()
raw_body_measurements_11_12 <- foreign::read.xport(bmx) %>% 
  janitor::clean_names()
raw_glycohemoglobin_11_12 <- foreign::read.xport(ghb) %>% 
  janitor::clean_names()

# 2013-2014
download.file("https://wwwn.cdc.gov/nchs/nhanes/2013-2014/DEMO_H.XPT",
              demo <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2013-2014/DIQ_H.XPT",
              diq <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2013-2014/BMX_H.XPT",
              bmx <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2013-2014/GHB_H.XPT",
              ghb <- tempfile(), mode="wb", quiet = TRUE)
raw_demographics_13_14 <- foreign::read.xport(demo) %>% 
  janitor::clean_names()
raw_survey_responses_13_14 <- foreign::read.xport(diq) %>% 
  janitor::clean_names()
raw_body_measurements_13_14 <- foreign::read.xport(bmx) %>% 
  janitor::clean_names()
raw_glycohemoglobin_13_14 <- foreign::read.xport(ghb) %>% 
  janitor::clean_names()

# 2015-2016
download.file("https://wwwn.cdc.gov/nchs/nhanes/2015-2016/DEMO_I.XPT",
              demo <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2015-2016/DIQ_I.XPT",
              diq <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2015-2016/BMX_I.XPT",
              bmx <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2015-2016/GHB_I.XPT",
              ghb <- tempfile(), mode="wb", quiet = TRUE)
raw_demographics_15_16 <- foreign::read.xport(demo) %>% 
  janitor::clean_names()
raw_survey_responses_15_16 <- foreign::read.xport(diq) %>% 
  janitor::clean_names()
raw_body_measurements_15_16 <- foreign::read.xport(bmx) %>% 
  janitor::clean_names()
raw_glycohemoglobin_15_16 <- foreign::read.xport(ghb) %>% 
  janitor::clean_names()

# 2017-2018
download.file("https://wwwn.cdc.gov/nchs/nhanes/2017-2018/DEMO_J.XPT",
              demo <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2017-2018/DIQ_J.XPT",
              diq <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2017-2018/BMX_J.XPT",
              bmx <- tempfile(), mode="wb", quiet = TRUE)
download.file("https://wwwn.cdc.gov/nchs/nhanes/2017-2018/GHB_J.XPT",
              ghb <- tempfile(), mode="wb", quiet = TRUE)
raw_demographics_17_18 <- foreign::read.xport(demo) %>% 
  janitor::clean_names()
raw_survey_responses_17_18 <- foreign::read.xport(diq) %>% 
  janitor::clean_names()
raw_body_measurements_17_18 <- foreign::read.xport(bmx) %>% 
  janitor::clean_names()
raw_glycohemoglobin_17_18 <- foreign::read.xport(ghb) %>% 
  janitor::clean_names()

# Demographics data
raw_demographics_all <- bind_rows(
  raw_demographics_11_12,
  raw_demographics_13_14,
  raw_demographics_15_16,
  raw_demographics_17_18
) %>%
  as_tibble()

# Survey data
raw_survey_responses_all <- bind_rows(
  raw_survey_responses_11_12,
  raw_survey_responses_13_14,
  raw_survey_responses_15_16,
  raw_survey_responses_17_18
) %>%
  as_tibble()

# Body measurements data
raw_body_measurements_all <- bind_rows(
  raw_body_measurements_11_12,
  raw_body_measurements_13_14,
  raw_body_measurements_15_16,
  raw_body_measurements_17_18
) %>%
  as_tibble()

# Glycohemoglobin data
raw_glycohemoglobin_all <- bind_rows(
  raw_glycohemoglobin_11_12,
  raw_glycohemoglobin_13_14,
  raw_glycohemoglobin_15_16,
  raw_glycohemoglobin_17_18
) %>%
  as_tibble()

# Join into one dataset and add outcome label
df <- raw_demographics_all %>%
  full_join(raw_survey_responses_all, by = "seqn") %>%
  full_join(raw_body_measurements_all, by = "seqn") %>%
  full_join(raw_glycohemoglobin_all, by = "seqn") %>%
  mutate(
    lbxgh = as.numeric(as.character((lbxgh))),
    diq010 = as.numeric(as.character((diq010))),
    a1c = cut(lbxgh,breaks=c(0,5.7,6.5,1000),right=FALSE),
    diabetes_diagnosis = case_when(
      diq010 %in% 1 ~ 1,
      diq010 %in% c(2,3,9) ~ 0,
      diq010 %in% 7 ~ as.numeric(NA),
    ),
    diabetes = diabetes_diagnosis,
    diabetes = if_else(a1c=="[6.5,1e+03)" &!is.na(a1c), 1, diabetes),
    diabetes = as.integer(diabetes),
    diabetes = if_else(diabetes == 1, TRUE, FALSE),
    # Normalize weights
    weights = wtmec2yr / sum(wtmec2yr),
    # Normalize race
    race = case_when(
      ridreth3 == 1 ~ "Hispanic",
      ridreth3 == 2 ~ "Hispanic",
      ridreth3 == 3 ~ "White",
      ridreth3 == 4 ~ "Black",
      ridreth3 == 6 ~ "Asian",
      ridreth3 == 7 ~ "Other"
    )
  )

df_simp <- df %>%
  select(diabetes, age = ridageyr, bmi = bmxbmi, race = race, weights) %>%
  filter(!is.na(diabetes), !is.na(bmi))

write.csv(df_simp, "G:/My Drive/PhD/Bias detection & mitigation/Thesis-fuzzy-rough-Uncertainty-Bias/Python scripts/diabetes.csv")
