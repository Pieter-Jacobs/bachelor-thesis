from imports import *
from classes.PreProcessorSBert import PreProcessorSBert
from classes.Evaluator import Evaluator
from classes.Trainer import Trainer
from classes.Querier import Querier
from classes.ActiveLearner import ActiveLearner
from classes.QuerierRandom import QuerierRandom
from classes.QuerierRET import QuerierRET
from classes.QuerierRECT import QuerierRECT
from classes.QuerierSUD import QuerierSUD

# Make results reproducible
SEED = 1815
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def choose_querier(config, model, iterator, trainer):
    """
    Initialise and return the chosen querier based on the configuration file
    
    Parameters:
    ----------
    config: DictConfig
        Hydra configuration dictionary containing all input command line and standard values of the different parameters
    model: torch.nn
        The language model used for classification
    iterator: torchtext.data.Iterator
        The iterator used to iterate through the unlabeled dataset for computing uncertainties
    trainer:
        Trainer used to retrain on the redundancy pool for the RET heuristic
    Returns:
    --------
    querier: Querier
        The type of Querier used for querying and labeling data
    """
    querier = None
    if 'heuristic' not in config:
        querier = Querier(
            model=model,
            iterator=iterator,
            query_function=config.query_function,
            T=config.parameters.T,
            Q=config.parameters.Q
        )
    elif config.heuristic.name == 'random':
        querier = QuerierRandom(
            model=model,
            iterator=iterator,
            query_function=config.query_function,
            T=config.parameters.T,
            Q=config.parameters.Q,
        )
    elif config.heuristic.name == 'rect':
        querier = QuerierRECT(
            model=model,
            iterator=iterator,
            query_function=config.query_function,
            T=config.parameters.T,
            Q=config.parameters.Q,
            pool_size=config.heuristic.pool_size,
            l=config.heuristic.l
        )
    elif config.heuristic.name == 'ret':
        querier = QuerierRET(
            model=model,
            iterator=iterator,
            trainer=trainer,
            query_function=config.query_function,
            T=config.parameters.T,
            Q=config.parameters.Q,
            pool_size=config.heuristic.pool_size
        )
    elif config.heuristic.name == 'sud':
        querier = QuerierSUD(
            model=model,
            iterator=iterator,
            query_function=config.query_function,
            T=config.parameters.T,
            Q=config.parameters.Q,
            pool_size=config.heuristic.pool_size,
            K=config.heuristic.K
        )
    return querier


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Initialises all necassary modules for the active learning application

    Parameters:
    -----------
    cfg: Dictconfig
        Hydra configuration dictionary containing all input command line and standard values of the different parameters
    """
    print("Loading dataset: " + cfg.dataset.name)
    seed_df = pd.read_csv(
        hydra.utils.get_original_cwd() + cfg.dataset.seed_path)
    unlabeled_df = pd.read_csv(
        hydra.utils.get_original_cwd() + cfg.dataset.unlabeled_path)
    val_df = pd.read_csv(hydra.utils.get_original_cwd() + cfg.dataset.val_path)
    test_df = pd.read_csv(
        hydra.utils.get_original_cwd() + cfg.dataset.test_path)
    preProcessor = PreProcessorSBert()
    seed_df = preProcessor.pre_process(
        df=seed_df,
        selected_columns=cfg.dataset.columns_to_select,
        remove_labels=False
    )
    unlabeled_df = preProcessor.pre_process(
        df=unlabeled_df,
        selected_columns=cfg.dataset.columns_to_select,
        remove_labels=True)
    val_df = preProcessor.pre_process(
        df=val_df,
        selected_columns=cfg.dataset.columns_to_select,
        remove_labels=False
    )
    test_df = preProcessor.pre_process(
        df=test_df,
        selected_columns=cfg.dataset.columns_to_select,
        remove_labels=False
    )
    fields = preProcessor.create_fields()
    seed_ds, unlabeled_ds = preProcessor.create_datasets(labeled_df=seed_df,
                                                         unlabeled_df=unlabeled_df, fields=fields, is_test=False)
    val_ds = preProcessor.create_datasets(
        labeled_df=val_df, fields=fields, is_test=True)[0]
    test_ds = preProcessor.create_datasets(
        labeled_df=test_df, fields=fields, is_test=True)[0]

    model = BertForSequenceClassification.from_pretrained(
        # Use the 12-layer BERT model, with an uncased vocab.
        "bert-base-multilingual-cased" if cfg.dataset.name == 'kvk' else "bert-base-uncased",
        num_labels=cfg.dataset.feature_size,
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=cfg.parameters.dropout,
        attention_probs_dropout_prob=cfg.parameters.dropout
    )
    torch.save(model.state_dict(), os.path.join(
        hydra.utils.get_original_cwd(), "saves" + os.path.sep + 'model.pkl'))   # Save the initial parameters of the model for retraining
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_iterator = data.BucketIterator(
        dataset=seed_ds,
        batch_size=cfg.parameters.batch_size,
        device=device
    )
    val_iterator = data.BucketIterator(
        dataset=val_ds,
        batch_size=cfg.parameters.batch_size,
        device=device
    )
    test_iterator = data.BucketIterator(
        dataset=test_ds,
        batch_size=cfg.parameters.batch_size,
        device=device
    )
    query_iterator = data.BucketIterator(
        dataset=unlabeled_ds,
        batch_size=int(len(unlabeled_ds)/100),
        sort_within_batch=False,
        sort=False,
        shuffle=False,
        device=device
    )
    evaluator = Evaluator(
        model=model,
        iterator=test_iterator,
        n_epochs=1,
        feature_size=cfg.dataset.feature_size,
        metric_file=cfg.metric_file
    )
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            eps=cfg.optimizer.eps),
        train_iterator=train_iterator,
        evaluator=evaluator,
        n_epochs=cfg.parameters.epochs,
        val_iterator=val_iterator)

    querier = choose_querier(
        config=cfg,
        model=model,
        trainer=trainer,
        iterator=query_iterator
    )
    activeLearner = ActiveLearner(
        model, trainer, querier, cfg.optimizer, cfg.metric_file)
    activeLearner.loop()


if __name__ == "__main__":
    main()
