import detect_char
import detect_line

# reconst 모듈로 넘겨줄 paragraph list를 생성
# 아래 구현은 이미지에서 왼쪽 상단 192 X 64 부분을 잘라
# 12개 char로 만들어 넘겨주는 임시 구현
def get_graphs(img):
    chars = [Char(imresize(img[0:32, 0+i*32:i*32+32], [32, 32])) for i in range(0, 6)]
    l1 = Line(chars)
    chars = [Char(imresize(img[32:64, 0+i*32:i*32+32], [32, 32])) for i in range(0, 6)]
    l2 = Line(chars)
    p = Paragraph([l1, l2])
    return [p]