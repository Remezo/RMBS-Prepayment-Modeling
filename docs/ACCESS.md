# Private access for classmates

GitHub and AWS are separate permission systems. A GitHub invitation grants access to this code; it does not let someone read S3.

## Recommended: individual identities in the owner's AWS account

Use your existing IAM Identity Center setup, if available, and assign each classmate an individual identity and a read-only permission set containing `classmate-read-only-policy.json`. Have each student authenticate with their own `sfld-class` profile. The policy grants reads only under `s3sfld/parquet/release47-monthly/`; it grants no upload, delete, EC2, IAM, or raw ZIP permissions. No long-lived access key or shared password is needed.

An administrator must provision these identities and assignments. These files do not create users, invite people, change trust relationships, or attach policies. If you do not already use Identity Center, choose the identity arrangement before issuing access.

## If students use their own AWS accounts

A policy in their account alone is insufficient. The owner must also approve access through a bucket policy naming each exact role, or create a read-only role with a trust policy naming those approved principals. Both sides may need permission to call `sts:AssumeRole`. Do not use a wildcard principal. Ask each student for their AWS role ARN, not secret keys.

If the dataset uses a customer-managed KMS key, readers also need `kms:Decrypt` on that exact key and permission in its key policy. SSE-S3 avoids that additional key-policy requirement. Preserve HTTPS, server-side encryption, and S3 Block Public Access.

## Authentication

For an administrator-provided Identity Center login:

```bash
aws configure sso --profile sfld-class
aws sso login --profile sfld-class
```

For a compatible individual AWS console identity, `aws login --profile sfld-class --region us-east-2` requires the separate `SignInLocalDevelopmentAccess` permission. It is an authentication permission, not permission to read the dataset.

Then run:

```bash
aws sts get-caller-identity --profile sfld-class
python scripts/classmate_sample.py --profile sfld-class --year 2026 --month 1
```

Do not send credentials or login cache files to anyone. Do not add them to GitHub. Instructors should grant the narrowest useful dataset scope, and revoke access when the course ends.

## Troubleshooting

- `NoCredentialsError`: sign in with your own configured profile.
- `AccessDenied`: confirm the dataset permission, cross-account trust if applicable, and KMS permissions if using SSE-KMS.
- Missing `_SUCCESS`: the dataset is not yet published; wait for the owner to confirm completion.
- `No published performance records`: select a month covered by the published cohorts and source release.
- Download cap reached: reduce the requested rows or deliberately raise `--max-download-mib`.
- On remote EC2, omit `--profile` to use its instance role.

For ongoing shared use, consider S3 access logging, CloudTrail data events, and CloudWatch metrics, accounting for their charges before enabling them. These settings are not changed by the conversion tool.

References: [AWS CLI authentication](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-authentication.html), [S3 security practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html), [S3 pricing](https://aws.amazon.com/s3/pricing/).
