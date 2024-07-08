import uuid
import tos

# TOS配置信息
ak = ""
sk = ""
endpoint = ""
region = ""
bucket_name = ""

def upload_file_to_tos(local_file_path):
    try:
        # 创建 TosClientV2 对象
        client = tos.TosClientV2(ak, sk, endpoint, region)
        
        # 生成唯一的 object_key
        file_name = local_file_path.split("/")[-1]
        object_key = str(uuid.uuid4()) + "." + file_name.split(".")[-1]
        
        # 将本地文件上传到目标桶中
        client.put_object_from_file(bucket_name, object_key, local_file_path)
        
        # 生成预签名的 URL
        presigned_url = client.generate_presigned_url('GET', bucket_name, object_key, ExpiresIn=86400)
        
        return presigned_url

    except tos.exceptions.TosClientError as e:
        print('fail with client error, message:{}, cause: {}'.format(e.message, e.cause))
        return None

    except tos.exceptions.TosServerError as e:
        print('fail with server error: {}'.format(str(e)))
        return None

    except Exception as e:
        print('fail with unknown error: {}'.format(e))
        return None
if __name__ == "__main__":
    # 示例调用
    local_file_path = "/Users/kevin/Desktop/juben.wav"
    url = upload_file_to_tos(local_file_path)
    print("Uploaded file URL:", url)