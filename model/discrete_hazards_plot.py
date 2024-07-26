import matplotlib.pyplot as plt
import numpy as np
import os

def plot_survival_probabilities(survival_scores_all, censor_all, survtime_all, case_id_all, num_cases=10, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_cases = min(num_cases, len(case_id_all))

    for i in range(num_cases):
        fig, ax = plt.subplots(figsize=(10, 5))

        survival_scores = survival_scores_all[i]
        survival_scores = np.append(survival_scores, survival_scores[-1])  # Extend line to last period
        survival_time = survtime_all[i]
        censor = censor_all[i]
        case_id = case_id_all[i]

        # Time bins for 20 periods, each representing half a year
        time_bins = np.arange(len(survival_scores) + 1) * 0.5  # Multiply by 0.5 to represent half years

        # Plot predicted survival probabilities
        ax.step(time_bins[:-1], survival_scores, where='post', label='Predicted Survival Probabilities')

        # Add vertical line for true survival time, properly adjusted to the 6-month period scale
        survival_years = survival_time / 365  # Convert days to years
        ax.axvline(x=survival_years, color='green' if censor else 'red', linestyle='--', 
                   label='True Survival Time (Last Visit - Censored)' if censor else 'True Survival Time (Death)')

        ax.set_title(f"Case ID: {case_id}")
        ax.set_xlabel('Years')
        ax.set_ylabel('Predicted Survival Probability')
        ax.legend()

        # Adjusting x-axis to show labels in years but represent them at correct intervals
        ax.set_xticks(np.arange(0, 10.5, 1))  # Set ticks every 1 year
        ax.set_xticklabels([f"{int(x)}" for x in np.arange(0, 10.5, 1)])  # Label ticks as full years

        # Save the figure
        fig_path = os.path.join(output_dir, f"Case_{case_id}.png")
        fig.savefig(fig_path)
        plt.close(fig)  # Close the figure to free up memory
        
        
def plot_survival_probabilities_quartiles(survival_scores_all, censor_all, survtime_all, case_id_all, num_cases=10, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert lists to numpy arrays, only if they are not empty and properly shaped
    if len(survival_scores_all) > 0:
        survival_scores_all = np.concatenate(survival_scores_all)
    if len(censor_all) > 0 and all(np.ndim(elem) > 0 for elem in censor_all):
        censor_all = np.concatenate(censor_all)
    if len(survtime_all) > 0 and all(np.ndim(elem) > 0 for elem in survtime_all):
        survtime_all = np.concatenate(survtime_all)
    if len(case_id_all) > 0 and all(np.ndim(elem) > 0 for elem in case_id_all):
        case_id_all = np.concatenate(case_id_all)

    # Ensure lengths match
    if len(survtime_all) != len(survival_scores_all):
        raise ValueError("The length of survtime_all must match the length of survival_scores_all")
    if len(survtime_all) != len(censor_all):
        raise ValueError("The length of survtime_all must match the length of censor_all")
    if len(survtime_all) != len(case_id_all):
        raise ValueError("The length of survtime_all must match the length of case_id_all")

    # Calculate average survival scores
    average_survival_scores = np.mean(survival_scores_all, axis=0)

    # Calculate the 1st quartile and 4th quartile indices
    sorted_indices = np.argsort(survtime_all)
    q1_indices = sorted_indices[:len(sorted_indices) // 4]
    q4_indices = sorted_indices[-len(sorted_indices) // 4:]

    # Calculate the average survival scores for the 1st and 4th quartiles
    average_q1_survival_scores = np.mean([survival_scores_all[i] for i in q1_indices], axis=0)
    average_q4_survival_scores = np.mean([survival_scores_all[i] for i in q4_indices], axis=0)

    # Extend time_bins for the average survival curves
    time_bins = np.arange(len(average_survival_scores)) * 0.5  # Multiply by 0.5 to represent half years

    # Extend the survival scores to the end of the 10th year
    extended_time_bins = np.append(time_bins, 10.0)
    average_survival_scores = np.append(average_survival_scores, average_survival_scores[-1])
    average_q1_survival_scores = np.append(average_q1_survival_scores, average_q1_survival_scores[-1])
    average_q4_survival_scores = np.append(average_q4_survival_scores, average_q4_survival_scores[-1])

    # Calculate mean survival time
    mean_survival_time = np.mean(survtime_all) / 365  # Convert days to years

    # Calculate mean survival times for the 1st and 4th quartiles
    mean_q1_survival_time = np.mean([survtime_all[i] for i in q1_indices]) / 365
    mean_q4_survival_time = np.mean([survtime_all[i] for i in q4_indices]) / 365

    # Debug statements
    print(f"extended_time_bins: {extended_time_bins.shape}")
    print(f"average_survival_scores: {average_survival_scores.shape}")
    print(f"average_q1_survival_scores: {average_q1_survival_scores.shape}")
    print(f"average_q4_survival_scores: {average_q4_survival_scores.shape}")

    # Plot survival curves for the averages
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot predicted survival probabilities for averages
    ax.step(extended_time_bins, average_survival_scores, where='post', label='Average Survival Probability', color='black')
    ax.step(extended_time_bins, average_q1_survival_scores, where='post', label='Average 1st Quartile Survival Probability', color='blue')
    ax.step(extended_time_bins, average_q4_survival_scores, where='post', label='Average 4th Quartile Survival Probability', color='red')

    # Add vertical lines for mean survival times
    ax.axvline(x=mean_survival_time, color='black', linestyle='--', label='Mean Survival Time')
    ax.axvline(x=mean_q1_survival_time, color='blue', linestyle='--', label='Mean 1st Quartile Survival Time')
    ax.axvline(x=mean_q4_survival_time, color='red', linestyle='--', label='Mean 4th Quartile Survival Time')

    ax.set_title("Average Survival Curves")
    ax.set_xlabel('Years')
    ax.set_ylabel('Predicted Survival Probability')
    ax.legend()

    # Adjusting x-axis to show labels in years but represent them at correct intervals
    ax.set_xticks(np.arange(0, 11, 1))  # Set ticks every 1 year
    ax.set_xticklabels([f"{int(x)}" for x in np.arange(0, 11, 1)])  # Label ticks as full years

    # Save the figure
    fig_path = os.path.join(output_dir, f"Average_Survival_Curves.png")
    fig.savefig(fig_path)
    plt.close(fig)  # Close the figure to free up memory
    
    
    
    
    
def plot_survival_probabilities_modalities(survival_vectors, censor_all, survtime_all, case_index, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define colors and labels for the modalities
    colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown']
    labels = ['Average (Late Fusion)','Clinical', 'Pathology Report', 'Gene Exp.', 'DNA Meth.', 'miRNA', 'Protein Exp.']

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, survival_scores_all in enumerate(survival_vectors):
        # Convert lists to numpy arrays, only if they are not empty and properly shaped
        #if len(survival_scores_all) > 0:
            #survival_scores_all = survival_scores_all#np.concatenate(survival_scores_all)

        # Extract data for the specific case
        survival_scores = survival_scores_all[case_index]

        # Extend time_bins for the survival curve
        time_bins = np.arange(len(survival_scores)) * 0.5  # Multiply by 0.5 to represent half years
        extended_time_bins = np.append(time_bins, 10.0)
        survival_scores = np.append(survival_scores, survival_scores[-1])

        # Plot the survival probability for the current modality
        ax.step(extended_time_bins, survival_scores, where='post', label=f'{labels[idx]} Survival Probability', color=colors[idx])

    # Extract censor and survival time for the specific case
    censor = censor_all[case_index]
    survtime = survtime_all[case_index]

    # Add vertical line for true survival time
    survival_years = survtime / 365  # Convert days to years
    ax.axvline(x=survival_years, color='black', linestyle='--', label='True Survival Time')

    ax.set_title(f"Survival Curves for Case {case_index}")
    ax.set_xlabel('Years')
    ax.set_ylabel('Predicted Survival Probability')
    ax.legend(loc='upper right')

    # Adjusting x-axis to show labels in years but represent them at correct intervals
    ax.set_xticks(np.arange(0, 11, 1))  # Set ticks every 1 year
    ax.set_xticklabels([f"{int(x)}" for x in np.arange(0, 11, 1)])  # Label ticks as full years

    # Save the figure
    fig_path = os.path.join(output_dir, f"Survival_Curves_Case_{case_index}.png")
    fig.savefig(fig_path)
    plt.close(fig)  # Close the figure to free up memory
