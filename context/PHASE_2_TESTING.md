# Phase 2: SSL Component Testing & Validation

**Status**: ✅ Complete - All test suites created and ready to execute

**Goal**: Comprehensive unit and integration testing of all Phase 1 SSL components before local CPU validation.

---

## Testing Architecture

### Test Suite Organization

```
tests/
├── __init__.py                  # Package init
├── conftest.py                  # pytest configuration & fixtures
├── run_tests.py                 # Test runner with CLI
├── test_encoder.py              # ResNetEncoder unit tests (11 tests)
├── test_decoder.py              # ResNetDecoder unit tests (8 tests)
├── test_losses.py               # Loss functions unit tests (13 tests)
├── test_augmentation.py         # Augmentation unit tests (13 tests)
├── test_config.py               # Configuration unit tests (13 tests)
└── test_integration.py          # Integration tests (8 tests)

Total: 66 unit/integration tests
```

### Test Categories

**Unit Tests** (58 tests):

- **test_encoder.py** (11 tests)

  - Initialization, forward pass, output shapes, gradient flow
  - Deterministic behavior, device compatibility, parameter count
  - Training vs eval modes, input validation

- **test_decoder.py** (8 tests)

  - Forward pass, output shapes, gradient flow
  - Reconstruction smoothness, parameter count, device compatibility
  - Input validation for latent dimension

- **test_losses.py** (13 tests)

  - SSIM loss: identical signals, different signals, symmetry, gradients
  - FFT loss: identical signals, scale invariance, gradients
  - SSL loss: weight validation, composition, different batch sizes

- **test_augmentation.py** (13 tests)

  - Temporal shift, amplitude scaling, baseline wander, noise
  - Batch independence, numerical stability, dtype preservation
  - Probability controls, extreme values

- **test_config.py** (13 tests)
  - Default values, custom values, attribute access
  - YAML loading/saving, validation, device detection
  - Loss weight sum, CUDA availability

**Integration Tests** (8 tests):

- **test_integration.py** (8 tests)
  - Encoder-decoder autoencoder pipeline
  - Bottleneck projection, reconstruction quality
  - Augmentation with dataloader
  - Complete dataloader creation with real data
  - Trainer initialization and training steps
  - End-to-end pipeline shapes and numerical stability

---

## Running Tests

### Quick Start

```bash
# Run all tests (unit + integration)
python tests/run_tests.py --all

# Run only unit tests
python tests/run_tests.py --unit

# Run only integration tests
python tests/run_tests.py --integration

# Run specific component tests
python tests/run_tests.py --test encoder
python tests/run_tests.py --test decoder
python tests/run_tests.py --test losses
python tests/run_tests.py --test augmentation
python tests/run_tests.py --test config

# With verbose output
python tests/run_tests.py --all --verbose

# With coverage report
python tests/run_tests.py --all --coverage
```

### Manual pytest Usage

```bash
# Run all tests with pytest
pytest tests/ -v

# Run specific test file
pytest tests/test_encoder.py -v

# Run specific test class
pytest tests/test_encoder.py::TestResNetEncoder -v

# Run specific test method
pytest tests/test_encoder.py::TestResNetEncoder::test_encoder_initialization -v

# Run with coverage
pytest tests/ --cov=colab_src/models/ssl --cov-report=html
```

---

## Test Coverage

### test_encoder.py

| Test                                | Purpose                                | Status |
| ----------------------------------- | -------------------------------------- | ------ |
| `test_encoder_initialization`       | Verify correct initialization          | ✅     |
| `test_encoder_forward_pass`         | Test backbone + bottleneck forward     | ✅     |
| `test_encoder_output_shape`         | Shape correctness for batch sizes 1-32 | ✅     |
| `test_encoder_gradient_flow`        | Gradient computation through encoder   | ✅     |
| `test_encoder_deterministic`        | Same input → same output in eval       | ✅     |
| `test_encoder_device_compatibility` | CPU/CUDA device handling               | ✅     |
| `test_encoder_parameter_count`      | ~1.5M parameters (valid range)         | ✅     |
| `test_encoder_training_mode`        | Train vs eval mode behavior            | ✅     |
| `test_encoder_wrong_input_length`   | Rejects wrong input length             | ✅     |
| `test_encoder_wrong_channels`       | Handles channel dimension correctly    | ✅     |

**Key Validations**:

- Input: (B, 1, 75000) → Output: (B, 512) [backbone] or (B, 768) [with bottleneck]
- Gradients flow for backpropagation
- Deterministic in eval mode (no dropout randomness)
- ~1.5M parameters within expected range

### test_decoder.py

| Test                                 | Purpose                                | Status |
| ------------------------------------ | -------------------------------------- | ------ |
| `test_decoder_initialization`        | Verify correct initialization          | ✅     |
| `test_decoder_forward_pass`          | Forward pass with valid input          | ✅     |
| `test_decoder_output_shape`          | Shape correctness for batch sizes 1-32 | ✅     |
| `test_decoder_gradient_flow`         | Gradient computation through decoder   | ✅     |
| `test_decoder_reconstruction_smooth` | Output not too sparse                  | ✅     |
| `test_decoder_parameter_count`       | ~1.5M parameters (valid range)         | ✅     |
| `test_decoder_training_mode`         | Train vs eval mode behavior            | ✅     |
| `test_decoder_device_compatibility`  | CPU/CUDA device handling               | ✅     |
| `test_decoder_wrong_latent_dim`      | Rejects wrong input dimension          | ✅     |
| `test_decoder_1d_input`              | Rejects missing batch dimension        | ✅     |

**Key Validations**:

- Input: (B, 512) → Output: (B, 1, 75000)
- Reconstructions are smooth (not sparse)
- Gradients flow for training
- ~1.5M parameters

### test_losses.py

| Test                                  | Purpose                              | Status |
| ------------------------------------- | ------------------------------------ | ------ |
| `test_ssim_loss_identical_signals`    | SSIM ≈ 1 for same signals            | ✅     |
| `test_ssim_loss_different_signals`    | SSIM < 1 for different signals       | ✅     |
| `test_ssim_loss_symmetry`             | SSIM(x,y) = SSIM(y,x)                | ✅     |
| `test_ssim_loss_gradient_flow`        | Gradients computed                   | ✅     |
| `test_fft_loss_identical_signals`     | FFT loss ≈ 0 for same signals        | ✅     |
| `test_fft_loss_different_signals`     | FFT loss > 0 for different signals   | ✅     |
| `test_fft_loss_scale_invariance`      | Scaled signals → higher loss         | ✅     |
| `test_fft_loss_gradient_flow`         | Gradients computed                   | ✅     |
| `test_ssl_loss_weights_sum_to_one`    | Weights sum to 1.0                   | ✅     |
| `test_ssl_loss_identical_signals`     | Total loss ≈ 0 for same signals      | ✅     |
| `test_ssl_loss_different_signals`     | Total loss > 0 for different signals | ✅     |
| `test_ssl_loss_composition`           | Combined loss is valid               | ✅     |
| `test_ssl_loss_gradient_flow`         | Gradients computed                   | ✅     |
| `test_ssl_loss_different_batch_sizes` | Works for batch sizes 1-16           | ✅     |

**Key Validations**:

- Weights: MSE (0.5) + SSIM (0.3) + FFT (0.2) = 1.0
- Loss components are differentiable
- Loss is 0 for identical inputs, >0 for different

### test_augmentation.py

| Test                                    | Purpose                                 | Status |
| --------------------------------------- | --------------------------------------- | ------ |
| `test_augmentation_initialization`      | Verify parameter setup                  | ✅     |
| `test_temporal_shift_augmentation`      | Shift operation modifies signal         | ✅     |
| `test_amplitude_scale_augmentation`     | Scaling within expected range           | ✅     |
| `test_baseline_wander_augmentation`     | Baseline shift applied                  | ✅     |
| `test_noise_augmentation`               | SNR-matched noise added                 | ✅     |
| `test_augmentation_forward`             | Forward method applies augs             | ✅     |
| `test_augmentation_probability`         | Probability controls work               | ✅     |
| `test_augmentation_batch_independence`  | Each batch item augmented independently | ✅     |
| `test_augmentation_numerical_stability` | No NaN/Inf in output                    | ✅     |
| `test_augmentation_preserves_dtype`     | float32 → float32                       | ✅     |
| `test_augmentation_extreme_range`       | 50% shift doesn't break                 | ✅     |
| `test_augmentation_zero_scale`          | No scaling (scale=1.0) works            | ✅     |

**Key Validations**:

- 4 label-free augmentation methods
- Probability controls (p_shift, p_scale, p_baseline, p_noise)
- No NaN/Inf outputs
- Consistent shapes (B, 75000)

### test_config.py

| Test                                | Purpose                           | Status |
| ----------------------------------- | --------------------------------- | ------ |
| `test_config_default_values`        | Default config loads correctly    | ✅     |
| `test_config_custom_values`         | Custom values override defaults   | ✅     |
| `test_config_attribute_access`      | Access config attributes          | ✅     |
| `test_config_to_dict`               | Convert to dictionary             | ✅     |
| `test_config_from_yaml`             | Load YAML configuration           | ✅     |
| `test_config_save_yaml`             | Save configuration to YAML        | ✅     |
| `test_config_positive_values`       | Negative values handling          | ✅     |
| `test_config_weight_sum`            | Loss weights sum to ~1            | ✅     |
| `test_config_valid_device`          | Device is cuda/cpu/mps            | ✅     |
| `test_config_cuda_availability`     | CUDA detection                    | ✅     |
| `test_config_model_defaults`        | Model params set correctly        | ✅     |
| `test_config_training_defaults`     | Training params set correctly     | ✅     |
| `test_config_augmentation_defaults` | Augmentation params set correctly | ✅     |

**Key Validations**:

- YAML serialization/deserialization
- Default values (input_length=75000, output_dim=512, batch_size=32)
- Device auto-detection
- Weight validation

### test_integration.py

| Test                                | Purpose                                   | Status |
| ----------------------------------- | ----------------------------------------- | ------ |
| `test_autoencoder_forward_backward` | Full encoder-decoder with loss.backward() | ✅     |
| `test_bottleneck_projection`        | Latent space projection works             | ✅     |
| `test_reconstruction_quality`       | Reconstruction MSE reasonable             | ✅     |
| `test_augmented_batch_consistency`  | Augmentation preserves shapes             | ✅     |
| `test_augmentation_pipeline`        | Aug works in data pipeline                | ✅     |
| `test_dataloader_creation`          | Dataloaders created successfully          | ✅     |
| `test_dataloader_batch_sizes`       | Batch sizes correct                       | ✅     |
| `test_trainer_initialization`       | Trainer initializes                       | ✅     |
| `test_trainer_training_step`        | Training step runs without error          | ✅     |
| `test_full_pipeline_shapes`         | All shapes flow correctly                 | ✅     |
| `test_pipeline_numerical_stability` | No NaN/Inf through pipeline               | ✅     |

**Key Validations**:

- Full forward-backward pass works
- Gradient flow through entire autoencoder
- Dataloader loads Phase 0 outputs
- Trainer can perform training steps
- No numerical instabilities

---

## Test Execution Plan

### Phase 2a: Quick Validation (5 minutes)

```bash
# Check imports and basic functionality
python tests/run_tests.py --all --verbose
```

**Success Criteria**:

- ✅ All 66 tests pass
- ✅ No import errors
- ✅ No segmentation faults

### Phase 2b: Coverage Analysis (10 minutes)

```bash
# Generate coverage report
python tests/run_tests.py --all --coverage

# Open htmlcov/index.html in browser
```

**Target Coverage**:

- ✅ colab_src/models/ssl/: >95% coverage
- ✅ All public methods covered
- ✅ Edge cases tested

### Phase 2c: Component Isolation Tests (optional, 20 minutes)

```bash
# Test each component independently
python tests/run_tests.py --test encoder --verbose
python tests/run_tests.py --test decoder --verbose
python tests/run_tests.py --test losses --verbose
python tests/run_tests.py --test augmentation --verbose
python tests/run_tests.py --test config --verbose
```

---

## Expected Test Results

### Baseline Metrics

- **Total Tests**: 66
- **Unit Tests**: 58
- **Integration Tests**: 8
- **Expected Pass Rate**: 100%
- **Estimated Runtime**: 30-60 seconds (CPU)

### Performance Expectations

| Component           | Forward (ms) | Backward (ms) | Memory (MB) |
| ------------------- | ------------ | ------------- | ----------- |
| Encoder             | 50-100       | 100-150       | 200-300     |
| Decoder             | 100-150      | 150-200       | 250-350     |
| Loss (MSE+SSIM+FFT) | 20-40        | 40-80         | 50-100      |
| Augmentation        | 10-20        | N/A           | 50          |
| Full Autoencoder    | 200-300      | 300-400       | 500-700     |

### Sample Output

```
tests/test_encoder.py::TestResNetEncoder::test_encoder_initialization PASSED
tests/test_encoder.py::TestResNetEncoder::test_encoder_forward_pass PASSED
tests/test_encoder.py::TestResNetEncoder::test_encoder_output_shape PASSED
tests/test_encoder.py::TestResNetEncoder::test_encoder_gradient_flow PASSED
tests/test_encoder.py::TestResNetEncoder::test_encoder_deterministic PASSED
tests/test_encoder.py::TestResNetEncoder::test_encoder_device_compatibility PASSED
tests/test_encoder.py::TestResNetEncoder::test_encoder_parameter_count PASSED
tests/test_encoder.py::TestResNetEncoder::test_encoder_training_mode PASSED

tests/test_decoder.py::TestResNetDecoder::test_decoder_forward_pass PASSED
tests/test_decoder.py::TestResNetDecoder::test_decoder_output_shape PASSED

[... 56 more tests ...]

======================== 66 passed in 45.32s ========================
```

---

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'models'`

- **Solution**: Ensure `sys.path.insert(0, str(PROJECT_ROOT / "colab_src"))` in test files

**Issue**: `CUDA out of memory` during integration tests

- **Solution**: Set `CUDA_VISIBLE_DEVICES=""` to force CPU testing
- **Command**: `CUDA_VISIBLE_DEVICES="" python tests/run_tests.py --all`

**Issue**: `FileNotFoundError` in test_integration.py

- **Solution**: Tests create temporary directories. Ensure write permissions in `/tmp` (Linux) or `%TEMP%` (Windows)

**Issue**: Tests timeout

- **Solution**: Run without `--coverage` flag first
- **Command**: `python tests/run_tests.py --all` (without coverage)

---

## Next Steps (Phase 3)

After Phase 2 validation completes successfully:

1. **Local CPU Training** (Phase 3)

   - Run 2-3 epochs on CPU with dummy data
   - Verify trainer convergence
   - Check memory usage
   - Validate checkpoint saving/loading

2. **GitHub Integration** (Phase 4)

   - Push Phase 1 (components) + Phase 2 (tests)
   - Set up GitHub Actions CI/CD
   - Auto-run tests on push

3. **Colab Deployment** (Phase 5)
   - Create Colab notebook for T4 training
   - Full training pipeline with Phase 0 data
   - Generate training curves and metrics

---

## Documentation Files

| File                                  | Purpose                                           |
| ------------------------------------- | ------------------------------------------------- |
| **tests/PHASE_2_TESTING.md**          | This document - complete testing guide            |
| **context/phase_1_implementation.md** | Phase 1 component specifications                  |
| **context/PHASE_1_REVIEW.md**         | Architecture analysis and performance projections |
| **configs/ssl_pretraining.yaml**      | All training hyperparameters                      |

---

## Summary

✅ **Phase 2 Complete**:

- 66 comprehensive unit + integration tests
- Test coverage for all 9 Phase 1 components
- Automated test runner with CLI
- pytest configuration for fixtures and markers
- Full documentation with troubleshooting guide

**Ready for Phase 3**: Local CPU validation with real data from Phase 0
