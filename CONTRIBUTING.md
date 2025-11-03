# Contributing to Narrow

Thank you for your interest in contributing to Narrow! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and constructive. We're all here to build something useful together.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Familiarity with Apache Arrow and linear algebra concepts

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/GeorgeLeePatterson/narrow.git
cd narrow

# Run tests to ensure everything works
cargo test

# Check formatting
cargo +nightly fmt --check

# Run lints
cargo clippy --all-features --all-targets
```

## Development Workflow

### 1. Create an Issue

Before starting work on a feature or bug fix:
- Check if an issue already exists
- If not, create one describing the problem or feature
- Discuss the approach before implementing

### 2. Fork and Branch

```bash
# Fork the repo on GitHub, then:
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

- Write clear, self-documenting code
- Add tests for new functionality
- Update documentation as needed
- Follow Rust best practices and idioms

### 4. Run CI Locally

Before pushing, ensure all checks pass:

```bash
# Format code
cargo +nightly fmt

# Run clippy
cargo clippy --all-features --all-targets -- -D warnings

# Run tests
cargo test --all-features

# Check documentation
cargo doc --all-features --no-deps
```

### 5. Commit and Push

```bash
# Use conventional commit format
git commit -m "feat: add new similarity metric"
# or
git commit -m "fix: correct dimension validation"

git push origin your-branch-name
```

### 6. Create Pull Request

- Provide a clear description of the changes
- Reference any related issues
- Ensure CI passes
- Request review

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `ci:` - CI/CD changes
- `chore:` - Maintenance tasks

## Code Style

### Rust Style

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` (nightly) for formatting
- Run `clippy` and address all warnings
- Maximum line length: 100 characters

### Documentation

- Document all public APIs with doc comments
- Include examples in doc comments where helpful
- Update README.md for user-facing changes

### Testing

- Write unit tests for all new functionality
- Add integration tests for end-to-end workflows
- Use `approx` crate for floating-point comparisons
- Aim for high code coverage

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority

- **SIMD Optimizations**: Optimize similarity metrics with SIMD
- **DataFusion Integration**: UDFs for vector operations in SQL
- **Sparse Vectors**: Implement `SparseVectorArray` type
- **Performance**: Benchmarking and optimization

### Medium Priority

- **Extended Operations**: More distance metrics, normalization
- **Float16 Support**: Add half-precision float support
- **Documentation**: Examples, tutorials, use case guides
- **Error Messages**: Improve error messages and diagnostics

### Always Welcome

- **Bug Fixes**: Fix any bugs you encounter
- **Tests**: Add more test coverage
- **Documentation**: Improve docs, fix typos
- **Examples**: Add real-world usage examples

## Design Principles

When contributing, keep these principles in mind:

1. **Safety First**: All unsafe operations must be encapsulated in safe APIs
2. **Zero-Copy**: Prefer views and borrows over copying data
3. **Type Safety**: Encode semantics in the type system where possible
4. **Arrow Native**: Leverage Arrow's columnar strengths
5. **Composable**: Ensure compatibility with DataFusion and linear algebra libraries

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: File an issue with a reproducible example
- **Features**: Open an issue to discuss before implementing
- **Chat**: [Create a discussion](https://github.com/GeorgeLeePatterson/narrow/discussions)

## Pull Request Review Process

1. **Automated Checks**: CI must pass (tests, lints, formatting)
2. **Code Review**: At least one maintainer approval required
3. **Testing**: Verify tests cover new functionality
4. **Documentation**: Check docs are updated
5. **Merge**: Squash merge to main branch

## Release Process

Releases are automated:

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md` (if exists)
3. Create and push a version tag: `git tag v0.1.0 && git push --tags`
4. GitHub Actions will build and publish to crates.io

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- Release notes (for significant contributions)
- Optional: Add yourself to `AUTHORS` or `CONTRIBUTORS` file

Thank you for contributing to Narrow! 🧮
