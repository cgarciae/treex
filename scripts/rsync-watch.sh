sshpass -p "cgarcia1234" rsync --exclude-from=".gitignore" -avzP . $1:~/code

# while inotifywait -r -e modify,create,delete . ; do
#     rsync --exclude-from=".gitignore" -avzP . $1:~/code
# done 