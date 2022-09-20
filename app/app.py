from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout



class MainApp(App):
    def build(self):
        self.title = "Vitural Assitant"
        self.main_layout = BoxLayout(orientation='vertical')

        

        self.btn = Button(text="Alo", size_hint=(1, 0.2))
        self.main_layout.add_widget(self.btn)

        return self.main_layout
        


if __name__ == '__main__':
    MainApp().run()