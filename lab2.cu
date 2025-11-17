#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <png.h>
// =============== OpenCV для визуализации ===============
#include <opencv2/opencv.hpp>

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
// Gaussian blur
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

void apply_gaussian_blur_on_cpu(const Image& input, Image& output) {
    const float gaussian_weights[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    const float weight_sum = 16.0f;

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            for (int channel = 0; channel < input.channel_count; ++channel) {
                float weighted_sum = 0.0f;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int neighbor_x = cpu_clamp_reflect_index(x + dx, input.width);
                        int neighbor_y = cpu_clamp_reflect_index(y + dy, input.height);
                        int pixel_index = (neighbor_y * input.width + neighbor_x) * input.channel_count + channel;
                        weighted_sum += input.pixel_data[pixel_index] * gaussian_weights[dy + 1][dx + 1];
                    }
                }
                int output_index = (y * input.width + x) * input.channel_count + channel;
                output.pixel_data[output_index] = static_cast<unsigned char>(
                    fminf(fmaxf(weighted_sum / weight_sum, 0.0f), 255.0f)
                );
            }
        }
    }
}

// ======================
// Sobel edge detection
// ======================

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

void apply_sobel_edge_on_cpu(const Image& input, Image& output) {
    const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            for (int channel = 0; channel < input.channel_count; ++channel) {
                int gradient_x = 0;
                int gradient_y = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int neighbor_x = cpu_clamp_reflect_index(x + dx, input.width);
                        int neighbor_y = cpu_clamp_reflect_index(y + dy, input.height);
                        int pixel_index = (neighbor_y * input.width + neighbor_x) * input.channel_count + channel;
                        unsigned char pixel_value = input.pixel_data[pixel_index];
                        gradient_x += pixel_value * sobel_x[dy + 1][dx + 1];
                        gradient_y += pixel_value * sobel_y[dy + 1][dx + 1];
                    }
                }
                float magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);
                magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);
                int output_index = (y * input.width + x) * input.channel_count + channel;
                output.pixel_data[output_index] = static_cast<unsigned char>(255 - magnitude);
            }
        }
    }
}

// ======================
// Median denoising
// ======================

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

void apply_median_filter_on_cpu(const Image& input, Image& output) {
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            for (int channel = 0; channel < input.channel_count; ++channel) {
                unsigned char neighborhood[9];
                int index = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int neighbor_x = cpu_clamp_reflect_index(x + dx, input.width);
                        int neighbor_y = cpu_clamp_reflect_index(y + dy, input.height);
                        neighborhood[index++] = input.pixel_data[(neighbor_y * input.width + neighbor_x) * input.channel_count + channel];
                    }
                }
                std::sort(neighborhood, neighborhood + 9);
                output.pixel_data[(y * input.width + x) * input.channel_count + channel] = neighborhood[4];
            }
        }
    }
}

// ======================
// PNG I/O
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
// Filter execution
// ======================

float execute_filter_on_gpu(const Image& input, Image& output, const std::string& filter_name) {
    size_t data_size = input.width * input.height * input.channel_count;
    unsigned char *device_input, *device_output;
    cudaMalloc(&device_input, data_size);
    cudaMalloc(&device_output, data_size);
    cudaMemcpy(device_input, input.pixel_data.data(), data_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size(
        (input.width + block_size.x - 1) / block_size.x,
        (input.height + block_size.y - 1) / block_size.y
    );

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event);

    if (filter_name == "blur") {
        apply_gaussian_blur_kernel<<<grid_size, block_size>>>(
            device_input, device_output, input.width, input.height, input.channel_count
        );
    } else if (filter_name == "edge") {
        apply_sobel_edge_kernel<<<grid_size, block_size>>>(
            device_input, device_output, input.width, input.height, input.channel_count
        );
    } else if (filter_name == "denoise") {
        apply_median_filter_kernel<<<grid_size, block_size>>>(
            device_input, device_output, input.width, input.height, input.channel_count
        );
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    output = Image(input.width, input.height, input.channel_count);
    cudaMemcpy(output.pixel_data.data(), device_output, data_size, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return milliseconds;
}

float execute_filter_on_cpu(const Image& input, Image& output, const std::string& filter_name) {
    auto start_time = std::chrono::high_resolution_clock::now();
    output = Image(input.width, input.height, input.channel_count);

    if (filter_name == "blur") {
        apply_gaussian_blur_on_cpu(input, output);
    } else if (filter_name == "edge") {
        apply_sobel_edge_on_cpu(input, output);
    } else if (filter_name == "denoise") {
        apply_median_filter_on_cpu(input, output);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end_time - start_time).count();
}

// =============== Визуализация через OpenCV (безопасная для headless) ===============
void show_results(const std::string& input_path) {
    // Проверяем, доступен ли графический дисплей (через переменную окружения)
    const char* display = std::getenv("DISPLAY");
    if (!display) {
        std::cout << "\n[INFO] No display found. Skipping image visualization.\n";
        std::cout << "You can view the output files manually:\n";
        std::cout << "  output_blur.png\n";
        std::cout << "  output_edge.png\n";
        std::cout << "  output_denoise.png\n";
        return;
    }

    // Загружаем изображения через OpenCV
    cv::Mat original = cv::imread(input_path);
    cv::Mat blur     = cv::imread("output_blur.png");
    cv::Mat edge     = cv::imread("output_edge.png");
    cv::Mat denoise  = cv::imread("output_denoise.png");

    if (original.empty()) {
        std::cerr << "Warning: Could not load original image for display.\n";
        return;
    }

    // Показываем окна
    cv::imshow("Original", original);
    cv::imshow("Gaussian Blur", blur);
    cv::imshow("Sobel Edge", edge);
    cv::imshow("Median Denoise", denoise);

    std::cout << "\nPress any key in any OpenCV window to exit...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// ======================
// Main program
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

    std::vector<std::string> filter_names = {"blur", "edge", "denoise"};
    std::vector<float> gpu_durations, cpu_durations;

    for (const std::string& filter : filter_names) {
        Image gpu_result, cpu_result;
        float gpu_time = execute_filter_on_gpu(source_image, gpu_result, filter);
        float cpu_time = execute_filter_on_cpu(source_image, cpu_result, filter);
        gpu_durations.push_back(gpu_time);
        cpu_durations.push_back(cpu_time);
        save_png_image("output_" + filter + ".png", gpu_result);
    }

    // Pretty table output
    std::cout << "\n=== FILTER PERFORMANCE (milliseconds) ===\n";
    std::cout << "+----------+----------+----------+----------+\n";
    std::cout << "| Filter   | GPU Time | CPU Time | Speedup  |\n";
    std::cout << "+----------+----------+----------+----------+\n";
    for (size_t i = 0; i < filter_names.size(); ++i) {
        float speedup = cpu_durations[i] / gpu_durations[i];
        printf("| %-8s | %8.2f | %8.2f | %7.2fx |\n",
               filter_names[i].c_str(),
               gpu_durations[i],
               cpu_durations[i],
               speedup);
    }
    std::cout << "+----------+----------+----------+----------+\n";

    return 0;
}
