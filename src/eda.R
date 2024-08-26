library(readr)
library(ggplot2)
library(dplyr)


# load
data <- read.csv("/Users/alexanderlorenz/Documents/GitHub/ai-fc/data/fbref/possession.csv")
cmp <- c("Joshua Kimmich","Leon Goretzka", "Thomas Müller", "Alphonso Davies")


# Carries_PrgDist: Progressive Carrying Distance

data %>%
  filter(Unnamed..0_level_0_Player %in% cmp ) %>% 
  ggplot(aes(x = Season, y = Carries_PrgDist, color = Unnamed..0_level_0_Player)) +
  geom_point() +
  geom_vline(xintercept = "2018-2019", linetype = "dashed", color = "red")
 
# Carries_1.3:  Carries into Final Third

data %>%
  filter(Unnamed..0_level_0_Player %in% cmp) %>% 
  ggplot(aes(x = Season, y = Carries_1.3, color = Unnamed..0_level_0_Player)) +
  geom_point() +
  geom_vline(xintercept = "2018-2019", linetype = "dashed", color = "grey")

# Plot Squad Stats vs Opponent Stats

cmp <- c("Squad Total", "Opponent Total")

data %>%
  filter(Unnamed..0_level_0_Player %in% cmp) %>%
  filter(Team == "Bayern Munich")%>% 
  ggplot(aes(x = Season, y = Carries_PrgDist, color = Unnamed..0_level_0_Player)) +
  geom_point() 


### Investigate Team Characteristics - Compare Squad Stats


# Compare Take Ons

cmp <- c("Bayern Munich", "Manchester City")

data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp )%>% 
  ggplot(aes(x = Season, y = Take.Ons_Att , color = Team)) +
  geom_point() +
  labs(title = "Take On Attempts")


data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp )%>% 
  ggplot(aes(x = Season, y = Take.Ons_Succ / Take.Ons_Att , color = Team)) +
  geom_point() +
  labs(title = "Take On Success Rate")

# Compare Take Ons

cmp <- c("Bayern Munich", "Manchester City")

data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp )%>% 
  ggplot(aes(x = Season, y = Take.Ons_Att , color = Team)) +
  geom_point() +
  labs(title = "Take On Attempts")

# Compare Touches in Attacking 1/3
data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp )%>% 
  ggplot(aes(x = Season, y = Touches_Att.3rd, color = Team)) +
  geom_point() +
  labs(title = "Touches in Attacking 1/3")

# Compare number of Successfull Passes Received
data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp )%>% 
  ggplot(aes(x = Season, y = Receiving_Rec, color = Team)) +
  geom_point() +
  labs(title = "Successfull Passes Received")


# Compare number of Successfull Passes Received
data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp )%>% 
  ggplot(aes(x = Season, y = Receiving_PrgR, color = Team)) +
  geom_point() +
  labs(title = "Successfull Progressive Passes Received")


### Investigate Coach Characteristics - Compare Squad Stats and see coach impact
cmp <- c("Bayern Munich",  "RB Leipzig")

data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp) %>%
  mutate(Season = factor(Season, ordered = TRUE)) %>% 
  ggplot(aes(x = Season, y = Touches_Mid.3rd, color = Team, group = Team)) +
  geom_line() + 
  geom_point() +  
  labs(title = "Touches in Mid 1/3")

# Take Ons - Number of attempts to take on defenders while dribbling
# Offensive indicator 
data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp) %>%
  mutate(Season = factor(Season, ordered = TRUE)) %>% 
  ggplot(aes(x = Season, y = Take.Ons_Att, color = Team, group = Team)) +
  geom_line() + 
  geom_point() +  
  labs(title = "Attempts on Take Ons")

data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp) %>%
  mutate(Season = factor(Season, ordered = TRUE)) %>% 
  ggplot(aes(x = Season, y = Take.Ons_Succ / Take.Ons_Att, color = Team, group = Team)) +
  geom_line() + 
  geom_point() +  
  labs(title = "Rate on Successfull Take Ons")

### Investigate Pressing by looking at defensive actions
data <- read.csv("/Users/alexanderlorenz/Documents/GitHub/ai-fc/data/fbref/defensive_actions.csv")

# List of columns to plot
columns_to_plot <- c("Tackles_Tkl", "Tackles_TklW", "Tackles_Def.3rd", "Tackles_Mid.3rd", 
                     "Tackles_Att.3rd", "Challenges_Tkl", "Challenges_Att", "Challenges_Tkl.", 
                     "Challenges_Lost", "Blocks_Blocks", "Blocks_Sh", "Blocks_Pass",
                     "Unnamed..17_level_0_Int","Unnamed..18_level_0_Tkl.Int","Unnamed..19_level_0_Clr","Unnamed..20_level_0_Err")



# For loop to iterate through columns and generate plots
for (column in columns_to_plot) {
  # Create plot for each column
  p <- data %>%
    filter(Unnamed..0_level_0_Player == "Squad Total") %>%
    filter(Team %in% cmp) %>%
    mutate(Season = factor(Season, ordered = TRUE)) %>%
    ggplot(aes_string(x = "Season", y = column, color = "Team", group = "Team")) +
    geom_line() + 
    geom_point() +
    labs(title = paste("Plot of", column))
  
  # Print each plot
  print(p)
}

# Challenges lost, Interceptions are the same

# IDEA: For same season, overlay data and look for correlarions. If they correlate maybe they have same
# coaching style


# Correlation in defending
library(reshape2)

m <- data %>%
  filter(Unnamed..0_level_0_Player == "Squad Total") %>%
  filter(Team %in% cmp) %>%
  mutate(Season = factor(Season, ordered = TRUE))

corr_matrix <- cor(m[, columns_to_plot])
melted_corr_mat <- melt(corr_matrix)

melted_corr_mat %>% 
  ggplot(aes(x = Var1, y = Var2, fill = value)) + 
  geom_tile() + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(x = "Variables", y = "Variables", title = "Correlation Heatmap")



