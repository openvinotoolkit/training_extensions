package u

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"
)

func DownloadFile(url, dst string) (int64, error) {
	out, err := os.Create(dst)
	if err != nil {
		log.Println("Create", err)
		return 0, err
	}
	defer out.Close()
	resp, err := http.Get(url)
	if err != nil {
		log.Println("Get", err)
		return 0, err
	}
	defer resp.Body.Close()

	nBytes, err := io.Copy(out, resp.Body)
	if err != nil {
		log.Println("Copy", err)
		return 0, err
	}
	return nBytes, nil
}

func AmqpServicesQueuesDelare(conn *rabbitmq.Connection, queuesNames []string) map[string]*amqp.Queue {
	servicesQueues := make(map[string]*amqp.Queue)

	ch, err := conn.Channel()
	if err != nil {
		fmt.Println("amqp channel", err)
	}
	defer ch.Close()

	for _, qName := range queuesNames {
		q, err := ch.QueueDeclare(
			qName, // name
			false, // durable
			false, // delete when unused
			false, // exclusive
			false, // no-wait
			nil,   // arguments
		)

		servicesQueues[qName] = &q
		if err != nil {
			fmt.Println("QueueDeclare", err)
		}
	}
	return servicesQueues
}

func FilterInt(ss []int, test func(int) bool) (ret []int) {
	for _, s := range ss {
		if test(s) {
			ret = append(ret, s)
		}
	}
	return
}

func StringToFolderName(s string) string {
	replacer := strings.NewReplacer(
		"/", " ",
		"<", " ",
		">", " ",
		":", " ",
		"\"", " ",
		"'", " ",
		"`", " ",
		"\\", " ",
		",", " ",
		".", " ",
		"\t", " ",
		"\n", " ",
		" ", " ",
	)
	s = replacer.Replace(s)
	s = strings.TrimSpace(s)
	underscores := regexp.MustCompile(`\s+`)
	s = underscores.ReplaceAllString(s, "_")
	return s
}
