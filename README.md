# GPU_MULTIGPU_TEST
Tensorflow1.4 (fake data) 
Tensorflow2.0 (Mnist)
Pytorch(Mnist)
NVLink Test on Pytorch (CIFAR10)


실행
-	[다운로드] <br>
https://github.com/ChanLIM/gpu_test 에서 ‘Clone or download’에서 zip파일로 다운로드. 
혹은 
git에서 직접 다운 받으려면 (git 설치 필요) <br>
(git 설치 : apt-get update && apt-get install git) <br>
git clone https://github.com/ChanLIM/gpu_test.git <br>
현재 폴더 하위에 ‘gpu_test’라는 폴더가 하나 생깁니다. 그 폴더로 들어간 후 프로그램 실행. <br>

-	[프로그램 실행] <br>
	sh run_test_NVLINK.sh<br>
	자동으로 딥러닝에 필요한 파일이 다운로드 되고 딥러닝이 시작됩니다.<br>
	(NVLink를 모니터링하려면 root 권한이 필요합니다. 맨 처음에 Password 입력을 요구합니다.)<br>
	기존의 run_test.sh도 수정하여 NVLINK테스트도 하도록 추가했습니다.<br>


-	[모니터링 프로그램 실행] <br>
	sh show_nvidia_status.sh<br>
	(기존에 있던 코드를 다소 수정하여, GPU간 통신 상태도 모니터링할 수 있도록 했습니다. <br> GPU간 NVLINK로 연결되어 있지 않아도 기존 기능은 똑같이 작동합니다.) 딥러닝 프로그램 실행 전, 후 아무 때나 실행하여도 작동합니다.<br>
