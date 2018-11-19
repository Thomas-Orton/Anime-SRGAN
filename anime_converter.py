from helper_libraries import create_high_res_video

def example():
    create_high_res_video("samples/AMG_test.mp4","samples/AMG_test_out.mp4")
    create_high_res_video("samples/LOGE1_test.mp4","samples/LOGE1_test_out.mp4")
    create_high_res_video("samples/VE_test.mp4","samples/VE_test_out.mp4")


if __name__ == "__main__":
    import sys
    file_in=sys.argv[1]
    file_out=sys.argv[2]
    create_high_res_video(file_in,file_out)
