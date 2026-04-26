# Training And System Explanation

## Project Positioning

This project is a football scouting recommendation system built around three practical tasks:

- Similar player recommendation
- Recruitment ranking
- Player profile visualization

The system combines:

- A feature-based baseline recommendation pipeline
- A learned embedding pipeline based on an autoencoder
- A Streamlit application for interactive exploration
- A data workflow that is gradually moving from local parquet files to BigQuery-based cloud integration

---

## Why A Baseline Is Necessary

Before using a learned model, the project first establishes a baseline recommendation system.

This is important for two reasons:

1. The baseline provides an interpretable reference.
2. It helps evaluate whether the learned embedding model really adds value beyond handcrafted features.

In football scouting, structured player statistics already contain meaningful information about style, role, and strengths. Because of that, feature-based similarity is not just a fallback method; it is a valid and useful recommendation approach on its own.

---

## What The Baseline Does

The baseline works directly in the handcrafted feature space.

### Input Features

The recommendation system uses 12 core features:

- `shots_p90`
- `xg_p90`
- `key_passes_p90`
- `pass_accuracy`
- `carries_p90`
- `dribble_success_rate`
- `interceptions_p90`
- `duel_win_rate`
- `recoveries_p90`
- `mean_x`
- `mean_y`
- `activity_spread`

These features cover four dimensions:

- Attacking output
- Ball progression
- Defensive ability
- Spatial behavior

### Baseline Logic

The baseline recommendation logic is implemented in:

- [src/recommender.py](src/recommender.py)

The process is:

1. Read the processed player feature matrix
2. Build similarity matrices
3. For a selected player, retrieve similarity scores against all candidates
4. Remove the player himself
5. Optionally filter to same-position players only
6. Return the Top-K most similar players

### Similarity Methods Used

Three methods are supported:

- Cosine similarity
- Euclidean similarity
- PCA + cosine similarity

### Why These Methods

- Cosine similarity captures directional similarity in player style
- Euclidean similarity captures overall distance in feature values
- PCA provides an alternative reduced feature space that may reduce redundancy

The baseline is therefore both practical and explainable.

---

## Why A Learned Model Is Still Needed

Although the baseline is useful, it works directly in the raw handcrafted feature space.

That creates a limitation:

- Player style may depend on hidden relationships between features
- Important patterns may exist in combinations of features rather than in single columns
- Raw similarity may not fully capture deeper role or style similarity

To address this, the project introduces a learned representation model: an autoencoder.

---

## Why Autoencoder Was Chosen

The goal of the learned model is not classification, but representation learning.

This project does not have direct supervision labels such as:

- Similar / not similar
- Good / bad player
- Transfer target / not target

Because of that, the task is better framed as unsupervised learning.

An autoencoder is suitable because:

- The data is tabular and structured
- The network can be lightweight
- The encoder can produce low-dimensional player embeddings
- The learned latent vectors can be reused for similarity search

---

## Autoencoder Structure

The model is defined in:

- [src/autoencoder.py](src/autoencoder.py)

### Architecture

- Encoder: `input_dim -> hidden_dim -> latent_dim`
- Decoder: `latent_dim -> hidden_dim -> input_dim`

### Current Configuration

- `input_dim = 12`
- `hidden_dim = 16`
- `latent_dim = 6`

### Why This Design

- 12 input dimensions match the selected core scouting features
- Hidden dimension 16 is large enough to learn non-linear compression but still lightweight
- Latent dimension 6 provides a balance between compression and information retention

The architecture is intentionally simple because the task involves medium-scale structured data rather than raw images or text.

---

## Training Logic

The training pipeline is implemented in:

- [src/train.py](src/train.py)

### Training Objective

The autoencoder learns to reconstruct the original player feature vector from a compressed latent representation.

That means:

- Input = player feature vector
- Target = the same player feature vector

This is standard self-reconstruction training.

### Data Pipeline

Training data is prepared by:

1. Selecting the 12 training features
2. Converting them to tensors
3. Creating a `TensorDataset(tensor_x, tensor_x)`
4. Splitting into training and validation sets

This makes the task fully unsupervised.

### Loss Function

- Mean Squared Error (`MSELoss`)

Why:

- The output is a continuous reconstructed feature vector
- MSE is the natural reconstruction loss for numeric tabular inputs

### Optimizer

- Adam

Why:

- Stable
- Efficient
- Works well for small-to-medium tabular neural networks

### Early Stopping

The training pipeline includes early stopping with patience.

Why:

- Prevents unnecessary training
- Reduces overfitting risk
- Saves the best model according to validation loss

---

## Preprocessing Design

This is one of the most important lessons from the project.

### Original Observation

At first, the local file `processed_features.parquet` trained successfully, but direct BigQuery training on `feature.v_dashboard_players` produced unstable behavior when sample size increased.

Later analysis showed that `processed_features.parquet` was not a direct export of the BigQuery table.

Instead, the notebook applied an additional preprocessing pipeline before saving the local training file:

1. `fillna(0)`
2. Z-score normalization within each `position_group`

### Why Position-Based Normalization Matters

The same feature value can mean very different things across positions.

For example:

- A forward and a center back should not be directly compared on `shots_p90`
- A winger and a midfielder may have very different distributions of `carries_p90`

Normalizing within `position_group` means the model learns relative style inside a role context, instead of mixing raw values across all positions.

### Current Training Preprocessing

The training pipeline now aligns with the notebook logic:

1. Convert feature columns to numeric
2. Fill missing values with 0
3. Apply z-score normalization grouped by `position_group`
4. Use safe fallbacks when group standard deviation is 0 or null

This improves consistency between:

- The local training pipeline
- The BigQuery training pipeline

---

## Training Optimization Decisions

The training pipeline has been improved in several ways.

### 1. BigQuery Training Support

Training no longer relies only on local parquet files.

It now supports:

- `local`
- `bigquery`

This allows the training process to consume the cloud feature engineering output directly.

### 2. Preprocessing Alignment

Instead of inventing a completely new cleaning strategy, the project aligns the BigQuery training pipeline with the original notebook preprocessing logic.

This avoids introducing unnecessary extra processing steps and preserves interpretability.

### 3. Training History Export

After training, the script now automatically exports:

- CSV loss history
- HTML interactive loss curve
- Optional PNG output when image export dependencies are available

This makes it easier to:

- Inspect convergence
- Compare runs
- Reuse results in reports and presentations

### 4. Gradual Scaling

Instead of directly training on the full dataset, the pipeline was validated progressively:

- Small sample first
- Then larger sample sizes
- Then troubleshooting preprocessing mismatches

This made debugging much easier and exposed the data-quality gap between local and BigQuery-based pipelines.

---

## How Embeddings Are Used

After the model is trained, the final recommendation does not use the decoder output.

Instead:

1. All players are passed through the encoder
2. The encoder output becomes a 6-dimensional embedding for each player
3. These embeddings are saved as artifacts
4. Cosine similarity is computed in the embedding space

This creates two recommendation routes:

- Baseline route: original processed feature space
- Learned route: autoencoder embedding space

These two routes are intentionally kept side by side for comparison.

---

## Streamlit Page Explanations

## 1. Similar Player Recommendation

File:

- [app/pages/Similar_Player.py](app/pages/Similar_Player.py)

### Purpose

This page compares:

- Baseline recommendation results
- Autoencoder embedding recommendation results

It is the main page for showing the recommendation logic of the project.

### Data Sources

Baseline feature input supports:

- Local processed feature parquet
- BigQuery feature table

Embedding input currently uses:

- Local embedding parquet artifact

### Logic

1. Load feature data
2. Load embedding table
3. Build baseline similarity matrices
4. Let the user select a player
5. Run both recommendation pipelines
6. Show Top-K results side by side
7. Show summary metrics such as overlap and position purity

### Why This Page Exists

This page makes it possible to compare handcrafted similarity and learned similarity in a single interface.

It answers:

- Who does baseline recommend?
- Who does the learned embedding recommend?
- Are the two models consistent?
- Does the learned representation produce a useful alternative search space?

---

## 2. Recruitment Ranking

File:

- [app/pages/Recruitment.py](app/pages/Recruitment.py)

### Purpose

This page focuses on a different business question:

Instead of asking “Who is most similar to a player?”, it asks:

“Given a target position and tactical preference, who is the best recruitment candidate?”

### Data Source

- Currently uses the local processed feature table

### Logic

1. Load feature data
2. Aggregate features into scouting dimensions:
   - attacking
   - progression
   - defensive
   - spatial
3. Let the user assign weights
4. Compute a weighted recruitment score
5. Rank players within the target position

### Why This Page Exists

Real scouting is not only about finding similar players.

It is also about:

- Prioritizing profiles
- Matching tactical needs
- Building a shortlist

This page turns the project from a similarity engine into a recruitment decision-support tool.

---

## 3. Player Profile Dashboard

File:

- [app/pages/Player_Profile_Dashboard.py](app/pages/Player_Profile_Dashboard.py)

### Purpose

This page is used to present a full player profile view:

- Basic information
- Market value history
- Player activity heatmaps

### Data Sources

Supports:

- `local`
- `gcs`
- `bigquery`

In BigQuery mode it queries:

- `feature.player_profile`
- `generaldata.market_value_history`
- `feature.player_heatmap`

### Logic

1. Load player options
2. Select a player
3. Query profile, market value history, and heatmap records by `player_key`
4. Render:
   - player information cards
   - value trend chart
   - value summary metrics
   - role-based heatmap layout

### Why This Page Exists

A recommendation system should not stop at generating names.

It should also help answer:

- Who is this player?
- How has his market value evolved?
- What areas of the pitch does he actually occupy?
- Why does he match the recommendation logic?

This page turns recommendation output into interpretable scouting context.

---

## Current System Value

The project is no longer just a single model experiment.

It now functions as a full scouting prototype with:

- Baseline recommendation
- Learned embedding recommendation
- Recruitment ranking
- Player profile visualization
- BigQuery-connected data flow

The strongest value of the project is not only the model itself, but the fact that:

- the recommendation logic is explainable
- the training pipeline is explicit
- the product has an interactive interface
- the data flow is moving toward a realistic cloud-connected workflow

---

## Future Work

- Expand training sample size further
- Improve consistency between feature tables and embedding artifacts
- Analyze similarity distributions more deeply
- Consider stronger representation learning approaches after the baseline pipeline is stabilized
- Deploy the Streamlit application to the cloud
- Add more realistic scouting constraints such as age, market value, contract status, and league strength
