import sagemaker

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = 'SageMaker IAM role'

# Define your training parameters and dataset locations
train_input = sagemaker.inputs.TrainingInput("S3 bucket location", content_type="csv")
output_path = "S3 bucket location"

# Initialize the estimator
estimator = sagemaker.estimator.Estimator(
    image_uri='Container Image URI',  # Replace with your Docker image URI
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=output_path,
    sagemaker_session=sagemaker_session, 
    # security_group_ids =  ['vpc security group id'],
    # subnets = ['vpc subnet id'],
    # enable_network_isolation=True,
    # training_repository_access_mode='Vpc',
    # training_image_config={'RepositoryAccessMode': 'Vpc'}
)

# Set hyperparameters
# estimator.set_hyperparameters(
#     latent_dim=100,
#     feature_size=3,
#     epochs=1000,
#     batch_size=16,
#     discriminator_steps=5,
#     gp_weight=10.0
# )

# Start training
# estimator.fit({'training_dataset': train_input})
# estimator.fit({train_input})
estimator.fit({'train': train_input})
