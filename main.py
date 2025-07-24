# Import libraries
import argparse
from statistics import mean
from models import run_model
from datasets import load_dataset
from fit_model import train_fitting
from dimension_reduction import run_dimension_reduction
from kl_based_dr import run_kl_based_dr_level
from kl_based_dr import run_kl_based_dr_weight
from scaling import scaling


def main():
    parser = argparse.ArgumentParser(description="Run dimension reduction techniques and prediction models on datasets")
    parser.add_argument('--dataset', type = str, required=True, help="Subset name")
    parser.add_argument('--model', type = str, required=True, choices=["logistic_regression", "random_forest", "gbc", "gaussian_nb"], help="Model algorithm name")
    parser.add_argument('--solver', type=str,  default="liblinear" ,
                        choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                        help="Solver for logistic regression")
    parser.add_argument('--penalty', type=str,  default="l1",
                        choices=["l1", "l2"],
                        help="Penalty for logistic regression")
    parser.add_argument('--test_size', type = float, default=0.25, help="Size of the test split (default: 0.25)")
    parser.add_argument('--random_state', type=int, default=1, help="Set a random seed for reproducibility (default: 1)")
    parser.add_argument('--risk_window_end', type = int, default=365, help="End of risk window outcome (default: 365)")
    parser.add_argument('--dimension_reduction', type = str, default=None, choices=['pca', 'kernel_pca', 'svd', 'auto_encoder', None], help = 'Dimension Reduction technique' )
    parser.add_argument('--number_components', type = int, default = 1, help = 'Number of components for Dimension reduction technique (default: 1)' )
    parser.add_argument('--scaling', type = str, default=None, choices=['standard_scaling', 'minmax_scaling', None], help='Apply scaling')
    parser.add_argument('--kl_based_level', action="store_true")
    parser.add_argument('--kl_based_weight', action="store_true")
    parser.add_argument('--number_runs', type = int, default=1, help="Number of times to run experiment (default: 1) ")
    parser.add_argument('--number_epochs', type = int, default=10, help="Number of epochs (default: 10) ")
    parser.add_argument('--number_batch', type = int, default=32, help="Size of batches (default: 32) ")
    parser.add_argument('--parent_weight', type=float, default=0.2, help="Parent weight is equal or higher than chosen value (default: 0.2) ")
    parser.add_argument('--child_weight', type=float, default=0.2, help="Child weight is lower than chosen value (default: 0.2) ")
    parser.add_argument('--chosen_level', type=int, default=5, help="Chosen level ")
    parser.add_argument('--chosen_window', type = str, default="365d", help="Chosen time window ")

    args = parser.parse_args()

    # For applying autoencoder only 1 run is possible
    if args.dimension_reduction == 'Autoencoder':
        print("For Autoencoder only 1 run possible, you can change number of epochs and batch size")
        args.number_runs = 1


    for i in range(args.number_runs):
        # Load the dataset
        print(f"Loading dataset: {args.dataset}")
        X_train, X_test, y_train, y_test, target = load_dataset(args.dataset, chosen_window=args.chosen_window, test_size=args.test_size, random_state=args.random_state, risk_window_end=args.risk_window_end)

        # Apply scaling
        if args.scaling:
            print(f'Applying scaling {args.scaling}')
            X_train, X_test = scaling(X_train, X_test, args.scaling)

        # Apply dimension reduction
        if args.dimension_reduction:
            print(f'Applying dimension reduction {args.dimension_reduction}')
            X_train, X_test = run_dimension_reduction(X_train, X_test, y_train, args.dimension_reduction, args.number_components, args.number_epochs, args.number_batch)

        # Apply line knowledge based dimension reduction
        if args.kl_based_level:
            print(f'Applying level knowledge based dimension reduction')
            X_train, X_test, y_train, y_test = run_kl_based_dr_level(args.dataset, X_train, X_test, args.chosen_level, risk_window_end=args.risk_window_end)

        # Apply weighted knowledge based dimension reduction
        if args.kl_based_weight:
            print(f'Applying weighted knowledge based dimension reduction')
            X_train, X_test, y_train, y_test = run_kl_based_dr_weight(args.dataset, X_train, X_test, args.parent_weight, args.child_weight, risk_window_end=args.risk_window_end)

        # Get the model
        print(f"Loading model: {args.model}")
        model = run_model(args.model, args.solver, args.penalty)

        # Train and fitting the model
        print("Training and fitting the model...")
        train_auc, test_auc, test_auprc = train_fitting(model, X_train, X_test, y_train, y_test, args.model)

    train_auc_scores = []
    test_auc_scores = []
    test_auprc_scores = []

    # Get average of multiple runs
    if args.number_runs > 1:
        train_auc_scores.append(train_auc)
        test_auc_scores.append(test_auc)
        test_auprc_scores.append(test_auprc)

        print('Average train AUC:', mean(train_auc_scores))
        print('Average test AUC:', mean(test_auc_scores))
        print('Average test AUPRC', mean(test_auprc_scores))


    else:
        print('Train AUC:', train_auc)
        print('Test AUC:', test_auc)
        print('Test AUPRC', test_auprc)


if __name__ == "__main__":
    main()