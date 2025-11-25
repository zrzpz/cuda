#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <png.h>
#include <opencv2/opencv.hpp>

// ======================
// CUDA error checking macro
// ======================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ======================
// Image structure
// ======================

struct Image {
    int width;
    int height;
    int channel_count;
    std::vector<unsigned char> pixel_data;

    Image() : width(0), height(0), channel_count(0) {}
    Image(int w, int h, int c) : width(w), height(h), channel_count(c) {
        pixel_data.resize(w * h * c);
    }
};

// ======================
// Boundary handling (reflect padding)
// ======================

__device__ __forceinline__ int reflect_index(int index, int limit) {
    if (index < 0) return -index;
    if (index >= limit) return 2 * limit - index - 2;
    return index;
}

__device__ __forceinline__ int clamp_reflect_index(int index, int limit) {
    if (limit <= 1) return 0;
    index = reflect_index(index, limit);
    while (index < 0 || index >= limit) {
        if (index < 0) index = -index;
        else index = 2 * limit - index - 2;
    }
    return index;
}

inline int cpu_clamp_reflect_index(int index, int limit) {
    if (limit <= 1) return 0;
    auto reflect = [](int v, int lim) {
        if (v < 0) return -v;
        if (v >= lim) return 2 * lim - v - 2;
        return v;
    };
    index = reflect(index, limit);
    while (index < 0 || index >= limit) {
        if (index < 0) index = -index;
        else index = 2 * limit - index - 2;
    }
    return index;
}

// ======================
// Kernels (same as original)
// ======================

__global__ void apply_gaussian_blur_kernel(
    const unsigned char* input_pixels,
    unsigned char* output_pixels,
    int image_width,
    int image_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image_width || y >= image_height) return;

    const float gaussian_weights[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    const float weight_sum = 16.0f;

    for (int channel = 0; channel < channels; ++channel) {
        float weighted_sum = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int neighbor_x = clamp_reflect_index(x + dx, image_width);
                int neighbor_y = clamp_reflect_index(y + dy, image_height);
                int pixel_index = (neighbor_y * image_width + neighbor_x) * channels + channel;
                int kernel_index = (dy + 1) * 3 + (dx + 1);
                weighted_sum += input_pixels[pixel_index] * gaussian_weights[kernel_index];
            }
        }
        int output_index = (y * image_width + x) * channels + channel;
        output_pixels[output_index] = static_cast<unsigned char>(
            fminf(fmaxf(weighted_sum / weight_sum, 0.0f), 255.0f)
        );
    }
}

__global__ void apply_sobel_edge_kernel(
    const unsigned char* input_pixels,
    unsigned char* output_pixels,
    int image_width,
    int image_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image_width || y >= image_height) return;

    const int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    for (int channel = 0; channel < channels; ++channel) {
        int gradient_x = 0;
        int gradient_y = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int neighbor_x = clamp_reflect_index(x + dx, image_width);
                int neighbor_y = clamp_reflect_index(y + dy, image_height);
                int pixel_index = (neighbor_y * image_width + neighbor_x) * channels + channel;
                unsigned char pixel_value = input_pixels[pixel_index];
                int kernel_index = (dy + 1) * 3 + (dx + 1);
                gradient_x += pixel_value * sobel_x[kernel_index];
                gradient_y += pixel_value * sobel_y[kernel_index];
            }
        }
        float magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);
        magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);
        int output_index = (y * image_width + x) * channels + channel;
        output_pixels[output_index] = static_cast<unsigned char>(255 - magnitude);
    }
}

__device__ void sort_nine_values(unsigned char* values) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8 - i; ++j) {
            if (values[j] > values[j + 1]) {
                unsigned char temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
}

__global__ void apply_median_filter_kernel(
    const unsigned char* input_pixels,
    unsigned char* output_pixels,
    int image_width,
    int image_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image_width || y >= image_height) return;

    for (int channel = 0; channel < channels; ++channel) {
        unsigned char neighborhood[9];
        int index = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int neighbor_x = clamp_reflect_index(x + dx, image_width);
                int neighbor_y = clamp_reflect_index(y + dy, image_height);
                neighborhood[index++] = input_pixels[(neighbor_y * image_width + neighbor_x) * channels + channel];
            }
        }
        sort_nine_values(neighborhood);
        output_pixels[(y * image_width + x) * channels + channel] = neighborhood[4];
    }
}

// ======================
// PNG I/O (unchanged)
// ======================

bool load_png_image(const std::string& filename, Image& image) {
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) return false;

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) { fclose(file); return false; }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_read_struct(&png_ptr, nullptr, nullptr); fclose(file); return false; }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        fclose(file);
        return false;
    }

    png_init_io(png_ptr, file);
    png_read_info(png_ptr, info_ptr);

    image.width = png_get_image_width(png_ptr, info_ptr);
    image.height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    if (bit_depth == 16) png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);
    image.channel_count = 4;
    image.pixel_data.resize(image.width * image.height * image.channel_count);

    std::vector<png_bytep> row_pointers(image.height);
    for (int y = 0; y < image.height; ++y) {
        row_pointers[y] = &image.pixel_data[y * image.width * image.channel_count];
    }

    png_read_image(png_ptr, row_pointers.data());
    png_read_end(png_ptr, nullptr);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(file);
    return true;
}

bool save_png_image(const std::string& filename, const Image& image) {
    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) return false;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) { fclose(file); return false; }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_write_struct(&png_ptr, nullptr); fclose(file); return false; }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(file);
        return false;
    }

    png_init_io(png_ptr, file);
    png_set_IHDR(png_ptr, info_ptr, image.width, image.height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    std::vector<png_bytep> row_pointers(image.height);
    for (int y = 0; y < image.height; ++y) {
        row_pointers[y] = const_cast<unsigned char*>(&image.pixel_data[y * image.width * image.channel_count]);
    }

    png_write_image(png_ptr, row_pointers.data());
    png_write_end(png_ptr, nullptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(file);
    return true;
}

// ======================
// Dual-GPU context
// ======================

struct DualGPUContext {
    int gpu0, gpu1;
    size_t split_height;
    int overlap;
};

DualGPUContext init_dual_gpu(int image_height) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Error: At least 2 GPUs required, but only " << device_count << " found.\n";
        exit(1);
    }

    DualGPUContext ctx;
    ctx.gpu0 = 0;
    ctx.gpu1 = 1;
    ctx.overlap = 1;
    ctx.split_height = image_height / 2;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, ctx.gpu0);
    std::cout << "GPU0: " << prop.name << "\n";
    cudaGetDeviceProperties(&prop, ctx.gpu1);
    std::cout << "GPU1: " << prop.name << "\n";

    return ctx;
}

// ======================
// Dual-GPU execution
// ======================

float execute_filter_on_gpu_dual(const Image& input, Image& output, const std::string& filter_name) {
    // Инициализация контекста для двух GPU: разбиение изображения по высоте с перекрытием
    DualGPUContext ctx = init_dual_gpu(input.height);

    size_t total_size = input.width * input.height * input.channel_count;
    output = Image(input.width, input.height, input.channel_count);

    // Вычисление размеров частей изображения с учётом перекрытия
    size_t height0 = ctx.split_height + ctx.overlap;
    size_t size0 = input.width * height0 * input.channel_count;
    size_t height1 = input.height - ctx.split_height + ctx.overlap;
    size_t size1 = input.width * height1 * input.channel_count;

    const unsigned char* h_input = input.pixel_data.data();
    unsigned char* h_output = output.pixel_data.data();

    // Регистрация хостовой памяти для ускоренного копирования (pinned memory)
    CUDA_CHECK(cudaHostRegister(const_cast<unsigned char*>(h_input), total_size, cudaHostRegisterReadOnly));
    CUDA_CHECK(cudaHostRegister(h_output, total_size, cudaHostRegisterDefault));

    // Указатели на видеопамять для двух GPU
    unsigned char *d_in0 = nullptr, *d_out0 = nullptr, *d_in1 = nullptr, *d_out1 = nullptr;

    // === Выделение видеопамяти на GPU0 ===
    CUDA_CHECK(cudaSetDevice(ctx.gpu0));
    CUDA_CHECK(cudaMalloc(&d_in0, size0));
    CUDA_CHECK(cudaMalloc(&d_out0, size0));

    // === Выделение видеопамяти на GPU1 ===
    CUDA_CHECK(cudaSetDevice(ctx.gpu1));
    CUDA_CHECK(cudaMalloc(&d_in1, size1));
    CUDA_CHECK(cudaMalloc(&d_out1, size1));

    // === Копирование данных на устройства ===
    // Верхняя часть (включая перекрытие) — на GPU0
    CUDA_CHECK(cudaMemcpy(d_in0, h_input, size0, cudaMemcpyHostToDevice));

    // Нижняя часть: начинается за (split_height - overlap) строк до границы
    const unsigned char* h_in1 = h_input + (ctx.split_height - ctx.overlap) * input.width * input.channel_count;
    CUDA_CHECK(cudaSetDevice(ctx.gpu1));
    CUDA_CHECK(cudaMemcpy(d_in1, h_in1, size1, cudaMemcpyHostToDevice));

    // Настройка размеров блоков и сеток для каждого GPU
    dim3 block(16, 16);
    dim3 grid0((input.width + block.x - 1) / block.x, (height0 + block.y - 1) / block.y);
    dim3 grid1((input.width + block.x - 1) / block.x, (height1 + block.y - 1) / block.y);

    // ========================
    // ПРОГРЕВ: запуск ядер на обоих GPU параллельно
    // ========================
    CUDA_CHECK(cudaSetDevice(ctx.gpu0));
    if (filter_name == "blur") {
        apply_gaussian_blur_kernel<<<grid0, block>>>(d_in0, d_out0, input.width, height0, input.channel_count);
    } else if (filter_name == "edge") {
        apply_sobel_edge_kernel<<<grid0, block>>>(d_in0, d_out0, input.width, height0, input.channel_count);
    } else if (filter_name == "denoise") {
        apply_median_filter_kernel<<<grid0, block>>>(d_in0, d_out0, input.width, height0, input.channel_count);
    }

    CUDA_CHECK(cudaSetDevice(ctx.gpu1));
    if (filter_name == "blur") {
        apply_gaussian_blur_kernel<<<grid1, block>>>(d_in1, d_out1, input.width, height1, input.channel_count);
    } else if (filter_name == "edge") {
        apply_sobel_edge_kernel<<<grid1, block>>>(d_in1, d_out1, input.width, height1, input.channel_count);
    } else if (filter_name == "denoise") {
        apply_median_filter_kernel<<<grid1, block>>>(d_in1, d_out1, input.width, height1, input.channel_count);
    }

    // Ядра запущены асинхронно — ожидаем завершения на всех устройствах
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================
    // ФАКТИЧЕСКОЕ ИЗМЕРЕНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ ЯДЕР
    // ========================
    cudaEvent_t start0, stop0, start1, stop1;
    float gpu0_ms = 0.0f, gpu1_ms = 0.0f;

    // --- Запуск и измерение времени на GPU0 ---
    CUDA_CHECK(cudaSetDevice(ctx.gpu0));
    CUDA_CHECK(cudaEventCreate(&start0));
    CUDA_CHECK(cudaEventCreate(&stop0));
    CUDA_CHECK(cudaEventRecord(start0));

    if (filter_name == "blur") {
        apply_gaussian_blur_kernel<<<grid0, block>>>(d_in0, d_out0, input.width, height0, input.channel_count);
    } else if (filter_name == "edge") {
        apply_sobel_edge_kernel<<<grid0, block>>>(d_in0, d_out0, input.width, height0, input.channel_count);
    } else if (filter_name == "denoise") {
        apply_median_filter_kernel<<<grid0, block>>>(d_in0, d_out0, input.width, height0, input.channel_count);
    }
    CUDA_CHECK(cudaEventRecord(stop0));

    // --- Запуск и измерение времени на GPU1 (сразу после GPU0 для минимизации задержек) ---
    CUDA_CHECK(cudaSetDevice(ctx.gpu1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventRecord(start1));

    if (filter_name == "blur") {
        apply_gaussian_blur_kernel<<<grid1, block>>>(d_in1, d_out1, input.width, height1, input.channel_count);
    } else if (filter_name == "edge") {
        apply_sobel_edge_kernel<<<grid1, block>>>(d_in1, d_out1, input.width, height1, input.channel_count);
    } else if (filter_name == "denoise") {
        apply_median_filter_kernel<<<grid1, block>>>(d_in1, d_out1, input.width, height1, input.channel_count);
    }
    CUDA_CHECK(cudaEventRecord(stop1));

    // Ожидание завершения всех операций на обоих GPU
    CUDA_CHECK(cudaDeviceSynchronize());

    // Получение времени выполнения для каждого GPU
    CUDA_CHECK(cudaEventElapsedTime(&gpu0_ms, start0, stop0));
    CUDA_CHECK(cudaEventElapsedTime(&gpu1_ms, start1, stop1));

    // Общее время — максимальное из двух, так как GPU работают параллельно
    float kernel_time = fmaxf(gpu0_ms, gpu1_ms);

    // ========================
    // СБОРКА РЕЗУЛЬТАТА: копирование частей обратно в хостовую память
    // ========================
    // Верхняя часть (без перекрытия) — напрямую из GPU0
    CUDA_CHECK(cudaMemcpy(
        h_output,
        d_out0,
        ctx.split_height * input.width * input.channel_count,
        cudaMemcpyDeviceToHost
    ));

    // Нижняя часть: пропускаем перекрывающиеся строки из результата GPU1
    unsigned char* h_out1 = h_output + ctx.split_height * input.width * input.channel_count;
    const unsigned char* d_out1_skip = d_out1 + ctx.overlap * input.width * input.channel_count;
    size_t out1_size = (input.height - ctx.split_height) * input.width * input.channel_count;
    CUDA_CHECK(cudaMemcpy(h_out1, d_out1_skip, out1_size, cudaMemcpyDeviceToHost));

    // ========================
    // ОСВОБОЖДЕНИЕ РЕСУРСОВ
    // ========================
    CUDA_CHECK(cudaFree(d_in0)); CUDA_CHECK(cudaFree(d_out0));
    CUDA_CHECK(cudaFree(d_in1)); CUDA_CHECK(cudaFree(d_out1));
    CUDA_CHECK(cudaEventDestroy(start0)); CUDA_CHECK(cudaEventDestroy(stop0));
    CUDA_CHECK(cudaEventDestroy(start1)); CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaHostUnregister(const_cast<unsigned char*>(h_input)));
    CUDA_CHECK(cudaHostUnregister(h_output));

    // Возвращаем время самого медленного GPU — это и есть реальное время обработки
    return kernel_time;
}

// ======================
// Visualization (unchanged)
// ======================

void show_results(const std::string& input_path) {
    const char* display = std::getenv("DISPLAY");
    if (!display) {
        std::cout << "\n[INFO] No display found. Skipping image visualization.\n";
        std::cout << "You can view the output files manually:\n";
        std::cout << "  output_blur.png\n";
        std::cout << "  output_edge.png\n";
        std::cout << "  output_denoise.png\n";
        return;
    }

    cv::Mat original = cv::imread(input_path);
    cv::Mat blur     = cv::imread("output_blur.png");
    cv::Mat edge     = cv::imread("output_edge.png");
    cv::Mat denoise  = cv::imread("output_denoise.png");

    if (original.empty()) {
        std::cerr << "Warning: Could not load original image for display.\n";
        return;
    }

    cv::imshow("Original", original);
    cv::imshow("Gaussian Blur", blur);
    cv::imshow("Sobel Edge", edge);
    cv::imshow("Median Denoise", denoise);

    std::cout << "\nPress any key in any OpenCV window to exit...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// ======================
// Main
// ======================

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image.png>\n";
        return 1;
    }

    Image source_image;
    if (!load_png_image(argv[1], source_image)) {
        std::cerr << "ERROR: Failed to load image: " << argv[1] << "\n";
        return 1;
    }

    std::cout << "Input image: " << source_image.width << "x" << source_image.height << "x" << source_image.channel_count << "\n";

    std::vector<std::string> filter_names = {"blur", "edge", "denoise"};
    std::vector<float> gpu_durations;

    for (const std::string& filter : filter_names) {
        Image gpu_result;
        float gpu_time = execute_filter_on_gpu_dual(source_image, gpu_result, filter);
        gpu_durations.push_back(gpu_time);
        save_png_image("output_" + filter + ".png", gpu_result);
    }

    std::cout << "\n=== FILTER PERFORMANCE ON DUAL GPU (milliseconds) ===\n";
    std::cout << "+----------+----------+\n";
    std::cout << "| Filter   | GPU Time |\n";
    std::cout << "+----------+----------+\n";
    for (size_t i = 0; i < filter_names.size(); ++i) {
        printf("| %-8s | %8.2f |\n",
               filter_names[i].c_str(),
               gpu_durations[i]);
    }
    std::cout << "+----------+----------+\n";

    return 0;
}
