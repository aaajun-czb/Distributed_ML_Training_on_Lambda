### Something about AWS

### Requirements
    1.文件的Permission
    直接复制文件进linux文件夹中（我在windows上编写代码的），会出现Others用户组的权限问题
    确保文件的权限全打开，在文件夹中通过命令chmod -R o+rX .
    赋予Others Read文件的权限

    2.S3的Permission
    需要为Function添加S3的策略
    我为了省事创建了一个全权限的策略，在IAM控制台为角色附加
    好像是可以在代码设置密钥的，但我没时间看了

    3.新image
    修改完代码重新push后，可以在Function的映像直接改映像

    4.Invoke的Permission
    需要为Function添加去Invoke其他Function的策略
    我直接为角色在IAM控制台添加了AllowAllInvoke的那个策略